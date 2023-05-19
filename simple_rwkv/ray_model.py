from io import BytesIO
from itertools import starmap
import json
import pickle
from typing import List
import time
from fastapi import Response
from fastapi.responses import StreamingResponse
import numpy as np
import requests
from starlette.requests import Request

import ray
from ray import serve
import torch
from simple_rwkv.lib_raven import get_model


class _RWKV():

    def __init__(self):
        self.model, self.pipeline = get_model()

    def _handle_batch(self, input):
        
        batch, state = zip(*input)
        b_shapes = [x.shape for x in batch]
        ('b_shapes', b_shapes)

        batch = self.batch_items(batch)
        state = self.batch_states(batch, state)

        out, state = self.model.forward(batch, state)
        state = list(zip(*state))
        
        if len(out.shape)==3:
            out = out.squeeze(0)
        out = out.unsqueeze(0) if len(out.shape)==0 else out.unsqueeze(1)

        # # print(state[0], '->>')
        # print(len(out), len(state), len(b_shapes))
        responses = zip(out, state, b_shapes)
        responses = starmap(lambda o, s, b_shape: (o if len(b_shape)==1 else o, s), responses)
        responses = list(responses)

        return responses
    
    def init_hidden(self, data):
        batch_size = 1 if len(data.shape)==1 else data.shape[0]
        args = self.model.args
        
        state = [None] * args.n_layer * 5
        for i in range(args.n_layer): # state: 0=att_xx 1=att_aa 2=att_bb 3=att_pp 4=ffn_xx
            dd = self.model.strategy[i]
            dev = dd.device
            atype = dd.atype
            state[i*5+0] = torch.zeros(batch_size, args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
            state[i*5+1] = torch.zeros(batch_size, args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous()
            state[i*5+2] = torch.zeros(batch_size, args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous()
            state[i*5+3] = torch.zeros(batch_size, args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous() - 1e30
            state[i*5+4] = torch.zeros(batch_size, args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
            state[i*5+0] = torch.zeros(batch_size, args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
            state[i*5+1] = torch.zeros(batch_size, args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous()
            state[i*5+2] = torch.zeros(batch_size, args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous()
            state[i*5+3] = torch.zeros(batch_size, args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous() - 1e30
            state[i*5+4] = torch.zeros(batch_size, args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
        return state
    
    def batch_states(self, batch, states):
        # print('st', len(batch))

        # make way for cat dimension if required
        states = [state or self.init_hidden(data) for state, data in zip(states, batch)]
        states = [[s.unsqueeze(0) if len(s.shape)==1 else s for s in state] for state in states]
        states = [torch.cat(state) for state in zip(*states)]
        # print('s0', states[0].shape)
        return states

    async def __call__(self, request: Request):
        raise NotImplementedError()
    
@serve.deployment()
class RWKVGenerate(_RWKV):
    @serve.batch(max_batch_size=128)
    async def handle_batch(self, input):
        return self._handle_batch(input)
    
    
    def batch_items(self, batch):
        # print('bi', len(batch))

        
        x, y, data = zip(*[(x,0,data) for x, data in enumerate(batch)])
        
        batch = torch.zeros(max(x)+1, max(y)+1).int() - 1
        batch[torch.tensor(x).int(),torch.tensor(y).int()] = torch.tensor(data).int()
        # print('bi_s', batch.shape)

        return batch



@serve.deployment()
class RWKVInfer(_RWKV):
    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.1)
    async def handle_batch(self, input):
        return self._handle_batch(input)
    
    def batch_items(self, batch):
        # print('bi', len(batch))

        x, y, data = zip(*[(x,y,data) for x, item in enumerate(batch) for y, data in enumerate(item.squeeze())])

        batch = torch.zeros(max(x)+1, max(y)+1).int() - 1
        batch[torch.tensor(x).int(),torch.tensor(y).int()] = torch.tensor(data).int()
        # print('bi_s', batch.shape)
        return batch

class RayRWKV():
            
    def __init__(self):
        self.infer = RWKVInfer.get_handle()
        self.generate = RWKVGenerate.get_handle()
        
    def forward(self, batch, state):
        # # print("________--__")
        # (batch.shape)
        # # print("__--________")

        data = ray.put((batch, state))
        
        if len(batch.shape)==1:
            # print('generate')
            response = self.generate.handle_batch.remote(data)
        else:
            # print('infer')
            response = self.infer.handle_batch.remote(data)

        # https://github.com/ray-project/ray/issues/22450
        # try:
        #     return ray.get(response)
        # finally:
        #     ray.cancel(response)
        out, state = ray.get(response)
        if len(batch.shape)==1 and len(out.shape)==3:
            # hacks and techdebt
            out = out.reshape(1, out.size(-1))
        # print(out.shape, len(state), torch.cat(state).shape)
        # print([s.shape for s in state])
        return out, state