
run_simple_ai_backend:
	python -m simple_rwkv

run_simple_ai_frontend:
	python -m simple_rwkv.simple_ai serve                                    

run_model:
	serve run ray_server:d

init_ray:
	ray start --head