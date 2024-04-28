import unittest
import pickle
import torch
import ray
from simple_rwkv.lib_raven import get_model
from simple_rwkv.ray_model import RayRWKV
from simple_rwkv.ray_model import RayRWKV, _RWKV, RWKVInfer, RWKVGenerate
from simple_rwkv.ray_server import d
from ray import serve
import concurrent.futures

generate_one = torch.tensor([1])
generate_one_batch = torch.tensor([[1]])
infer_one = torch.tensor([[1, 23, 4364, 2432]])
infer_batch = torch.tensor([[1, 23, 4364, 2432], [553, 1416, -1, -1]])


class TestRayRWKV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init()
        cls.s = serve.start(detached=False)
        serve.run(d)
        cls.model = RayRWKV()

    def test_generate_one_without_state(self):
        one_generate_input = torch.tensor([5])
        state = None

        out = self.model.forward(one_generate_input, state)

        # Add assertions or checks based on the expected result
        # Example assertion:
        self.assertEqual(len(out), 1)

    def test_generate_one_with_state(self):
        one_generate_input = torch.tensor([5])

        _, state = self.model.forward(one_generate_input, None)

        out, state = self.model.forward(one_generate_input, state)

        # Add assertions or checks based on the expected result
        # Example assertion:
        self.assertEqual(len(out), 1)

    def test_generate_single_batched(self):
        one_generate_input = torch.tensor([[5]])

        _, state = self.model.forward(one_generate_input, None)

        out, state = self.model.forward(one_generate_input, state)

        # Add assertions or checks based on the expected result
        # Example assertion:
        self.assertEqual(len(out), 1)

    def test_generate_state_has_effect(self):
        one_generate_input = torch.tensor([5])

        _, state = self.model.forward(one_generate_input, None)

        out_1, state = self.model.forward(one_generate_input, state)
        state = [s+1 for s in state]
        out_2, state = self.model.forward(one_generate_input, state)

        # Add assertions or checks based on the expected result
        # Example assertion:
        self.assertFalse((out_1==out_2).all())

    def test_infer_one_without_state(self):
        one_input = torch.tensor([[1, 23, 4364, 2432]])
        state = None

        out, state = self.model.forward(one_input, state)

        # Add assertions or checks based on the expected result
        # Example assertion:
        self.assertEqual(len(out), 1)

    def test_infer_one_with_state(self):
        one_input = torch.tensor(one_input)
        _, state = self.model.forward(one_input, None)  # Replace with the appropriate state

        out, state = self.model.forward(one_input, state)

        # Add assertions or checks based on the expected result
        # Example assertion:
        self.assertEqual(len(out), 1)

    def test_infer_batch_without_state(self):
        batch_input = torch.tensor([[1, 23, 4364, 2432], [553, 1416, -1, -1]])
        state = None

        out, state = self.model.forward(batch_input, state)

        # Add assertions or checks based on the expected result
        # Example assertion:
        self.assertEqual(len(out), batch_input.size(0))

    def test_infer_batch_with_state(self):
        batch_input = torch.tensor([[1, 23, 4364, 2432], [553, 1416, -1, -1]])
        state = get_model()  # Replace with the appropriate state

        out, state = self.model.forward(batch_input, state)

        # Add assertions or checks based on the expected result
        # Example assertion:
        self.assertEqual(len(out), batch_input.size(0))

    def test_infer_batch_with_state_ray_batch(self):
        pass


    def test_infer_one_ray_batch(self):
        """
        test batching using ray, this is achieved via multiple concurrent requests to the 

        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(self.model.forward, infer_one, None) for _ in range(160)]  # Submit tasks to the executor
            out = [result.result() for result in concurrent.futures.as_completed(results)]  # Retrieve results as they complete

        # Process the 'out' variable as needed

    def test_infer_batch_ray_batch(self):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(self.model.forward, infer_batch, None) for _ in range(10)]  # Submit tasks to the executor
            out = [result.result() for result in concurrent.futures.as_completed(results)]  # Retrieve results as they complete
    
    def test_generate_single_ray_batch(self):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(self.model.forward, generate_one, None) for _ in range(10)]  # Submit tasks to the executor
            out = [result.result() for result in concurrent.futures.as_completed(results)]  # Retrieve results as they complete
        # Process the 'out' variable as needed

if __name__ == '__main__':
    unittest.main()
