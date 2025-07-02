# Makefile per Maximum Flow Tabu Search

.PHONY: install test run clean sample

# Installazione
install:
	pip install -r requirements.txt

# Crea network di esempio
sample:
	python create_sample_network.py

# Test rapido
test: sample
	python main.py --network data/networks/network_sample.txt

# Esperimenti multipli
experiments: sample
	python main.py --network data/networks/network_sample.txt --multiple

# Batch su tutti i network
batch:
	python run_experiments.py

# Pulizia
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -rf data/results/*
	
# Setup completo
setup: install sample test
	@echo "✅ Setup completato! Usa 'make experiments' per iniziare."
