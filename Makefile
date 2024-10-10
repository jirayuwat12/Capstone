format all: 
	isort .
	black . -l 120 --exclude src/T2M-GPT
