PYTHON ?= python

.PHONY: help check quickstart reproduce verify test notebook smoke plan

help:
	@printf '%s\n' 'check: test and reproduce the supplied results and notebook' 'quickstart: train and evaluate a small CPU example (requires PyTorch)' 'reproduce: regenerate paper figures' 'verify: regenerate and compare with supplied results' 'test: run unit tests' 'notebook: save an executed analysis notebook under outputs/' 'smoke: test a small FQI fit and resume (requires PyTorch)' 'plan: list experiment tasks'

# Run in order: the notebook also writes figures to outputs/.
check:
	$(MAKE) test
	$(MAKE) verify
	$(MAKE) notebook

quickstart:
	$(PYTHON) examples/quickstart_example.py

reproduce:
	$(PYTHON) scripts/generate_simulation_figures.py
verify: reproduce
	$(PYTHON) scripts/verify_results.py
test:
	$(PYTHON) -m unittest discover -s tests -v
notebook:
	$(PYTHON) scripts/run_notebook.py
smoke:
	$(PYTHON) -m functional_fitted_q smoke
plan:
	$(PYTHON) scripts/plan_experiments.py
