
run_simple_ai_backend:
	python -m simple_rwkv

run_simple_ai_frontend:
	python obsidian_serve.py serve

run_model:
	serve run simple_rwkv.ray_server:d

init_ray:
	ray start --head