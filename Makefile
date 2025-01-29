format all: 
	isort . --skip src/T2M-GPT --skip src/BIN
	black . -l 120 --exclude src/T2M-GPT --exclude src/BIN