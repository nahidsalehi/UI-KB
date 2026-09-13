PYTHON ?= python
DEVICE ?= cpu

check:
	$(PYTHON) scripts/check_repository.py
	$(PYTHON) -m unittest discover -s tests -v

validate-dev:
	$(PYTHON) benchmarks/validate_benchmark.py --profile baseline --split dev --models e5,bge_m3,qwen3,matina

validate-test:
	$(PYTHON) benchmarks/validate_benchmark.py --profile baseline --split test --models e5,bge_m3,qwen3,matina

hazm-check:
	$(PYTHON) benchmarks/validate_benchmark.py --profile hazm --split legacy-dev130 --models e5,bge_m3,qwen3

reproduce-paper:
	DEVICE=$(DEVICE) bash scripts/reproduce_paper.sh

reproduce-hazm:
	DEVICE=$(DEVICE) bash scripts/reproduce_hazm_ablation.sh
