import unittest
import pickle
import torch
import ray
from simple_rwkv.lib_raven import get_model
from simple_rwkv.ray_model import RayRWKV
from simple_rwkv.ray_model import RayRWKV, _RWKV, RWKVInfer, RWKVGenerate
from simple_rwkv.ray_server import d
from ray import serve
from rwkv.utils import PIPELINE_ARGS, PIPELINE
from simple_rwkv import lib_raven

class TestRWKV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.infer = RWKVInfer.func_or_class()
        cls.generate = RWKVGenerate.func_or_class()



    def test_generate_with_state(self):
        generate_one_input = torch.tensor([1])

        initial_state = self.generate.init_hidden(generate_one_input)  # Replace with the appropriate state

        # Test one input with state
        data = [(generate_one_input, initial_state)]
        (out, state), = self.generate._handle_batch(data)
        self.assertFalse(all([(s0==s1).all() for s0, s1 in zip(initial_state, state)]), "state was not updated")

        # Add assertions or checks based on the expected result


    def test_infer_with_state(self):
        one_input = torch.tensor([[1, 23, 4364, 2432]])

        initial_state = self.infer.init_hidden(one_input)  # Replace with the appropriate state

        # Test one input with state
        data = [(one_input, initial_state)]
        (out, state), = self.infer._handle_batch(data)
        self.assertFalse(all([(s0==s1).all() for s0, s1 in zip(initial_state, state)]), "state was not updated")

        # Add assertions or checks based on the expected result

    def test_infer_without_state(self):
        one_input = torch.tensor([[1, 23, 4364, 2432]])

        infer = RWKVInfer.func_or_class()

        # Test one input without state
        data = [(one_input, None)]
        (out, state), = infer._handle_batch(data)
        self.assertFalse(all([(s==0).all() for s in state]), "state was not updated")
        # Add assertions or checks based on the expected result

examples = [
            'What is the answer to life, the univer and everything?',
            'Tell me about Jack Black!',
            'Write the python function to calculate pi'
        ]
class RWKVCompare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.s = serve.start(detached=False)
        serve.run(d)

        cls.model, pipeline = get_model()
        args = PIPELINE_ARGS(temperature=1e-1,top_p=0)
        source_string = 'hello, how are you today?'
        target_string = ''.join(list(pipeline.igenerate(source_string, token_count=10, args=args)))
        cls.source = pipeline.encode(source_string)
        cls.target = pipeline.encode(target_string)
        cls.pipeline = pipeline
        cls.args = args

        cls.ray_model, cls.ray_pipe = lib_raven.get_model(use_ray=True)

        pipeline.igenerate('hello, how are you?')
    
    def test_ray_equals_local_pipeline_generate(self):

        input_string = 'the answer is 42, what is the question?'
        ray_result = ''.join(self.ray_pipe.igenerate(input_string, args=self.args, token_count=10))
        local_result = ''.join(self.pipeline.igenerate(input_string, args=self.args, token_count=10))
        (ray_result, local_result)
        self.assertEquals(ray_result, local_result)

    def test_ray_equals_local_embedding_single_string(self):

        input_string = 'the answer is 42, what is the question?'
        ray_result = self.ray_pipe.infer(input_string, args=self.args)
        local_result = self.pipeline.infer(input_string, args=self.args)
        ([r.shape for r in ray_result])
        ([r.shape for r in local_result])
        self.assertTrue((torch.cat(ray_result)==torch.cat(local_result)).all())

    def test_ray_equals_local_embedding_single_list(self):

        input_string = examples[0]
        ray_result = self.ray_pipe.infer(input_string, args=self.args)
        local_result = self.pipeline.infer(input_string, args=self.args)
        ([r.shape for r in ray_result])
        ([r.shape for r in local_result])
        self.assertTrue((torch.cat(ray_result)==torch.cat(local_result)).all())
    
    def test_ray_equals_local_embedding_multi_list(self):

        ray_result = self.ray_pipe.infer(examples, args=self.args)
        local_result = self.pipeline.infer(examples, args=self.args)
        ([r.shape for r in ray_result])
        ([r.shape for r in local_result])
        self.assertTrue((torch.cat(ray_result)==torch.cat(local_result)).all())

if __name__ == '__main__':
    unittest.main()
