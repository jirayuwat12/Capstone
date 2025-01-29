format all: 
	isort . --skip src/T2M-GPT
	black . -l 120 --exclude src/T2M-GPT
