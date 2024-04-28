import json
import logging
import re
from dataclasses import dataclass

from simple_rwkv import lib_raven
import torch
from simple_ai.api.grpc.chat.server import LanguageModel

import logging

logger = logging.getLogger(__name__)

def format_chat_log(chat: list[dict[str, str]] = dict()) -> str:
    raw_chat_text = ""
    for item in chat:
        if item["role"] not in ("user", "assistant"):
            continue
        role = "Bob" if item.get("role") == "user" else "Alice"
        content = item.get("content").strip()
        content = re.sub("\n+", "\n", content)

        raw_chat_text += f"{role}: {content}\n\n"
    return raw_chat_text + "Alice:"


class RavenRWKVModel(LanguageModel):

    def __init__(self, cfg):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model, self.pipeline = lib_raven.get_model(cfg)

    

    def chat(
        self,
        chatlog: list[list[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.9,
        top_p: int = 0.5,
        presencePenalty: int = 0.4,
        countPenalty: int = 0.4,
        *args,
        **kwargs,
    ) -> str:
        logger.debug('Starting RavenRWKVModel chat...')

        prompt = format_chat_log(chatlog)
        output = lib_raven.complete(
            prompt,
            self.model,
            self.pipeline,
            prompt=None,
            token_count=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
        )

        output = "".join(output)

        return [{"role": "raven", "content": output}]

    def complete(
        self,
        *args,
        **kwargs,
    ) -> str:
        output = self.stream_complete(*args, **kwargs)
        output = "".join(output)
        logger.debug('Starting RavenRWKVModel complete...')

        return output

    def stream_complete(
        self,
        prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.9,
        top_p: int = 0.5,
        presencePenalty: int = 0.4,
        countPenalty: int = 0.4,
        stop=None,
        # *args,
        **kwargs,
    ) -> str:
        
        logger.debug('Starting RavenRWKVModel stream_complete...')
        stop = stop or None
        if stop:
            stop = json.loads(stop)
        output = lib_raven.complete(
            prompt,
            self.model,
            self.pipeline,
            token_count=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
            stop_words=stop,
        )
        yield from output

    def stream(
        self,
        chatlog: list[list[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.9,
        top_p: int = 0.5,
        presencePenalty: int = 0.4,
        countPenalty: int = 0.4,
        *args,
        **kwargs,
    ):
        logger.debug('Starting RavenRWKVModel stream...')
        
        yield [{"role": "raven"}]

        stop_words = ["\n\nBob:", "\n\nAlice:"]

        prompt = format_chat_log(chatlog)
        first = True
        for delta in lib_raven.complete(
            prompt,
            self.model,
            self.pipeline,
            prompt=None,
            token_count=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
            stop_words=stop_words,
        ):
            clean_delta = delta
            if first:
                clean_delta = delta[1:]  ## remove leading whitespace in completion
                first = False
            yield [{"content": clean_delta}]

    def embed(
        self,
        inputs: list = [],
    ) -> list:
        logger.debug('Starting RavenRWKVModel embed...')

        embeddings = lib_raven.embedding(inputs, self.model, self.pipeline)

        return embeddings.tolist()
