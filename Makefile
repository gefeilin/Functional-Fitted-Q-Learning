PYTHON ?= python

.PHONY: help reproduce verify test notebook smoke plan

help:
	@printf '%s\n' 'reproduce: regenerate paper figures' 'verify: compare with supplied results' 'test: run unit tests' 'notebook: run the analysis notebook' 'smoke: test a small FQI fit and resume' 'plan: list experiment tasks'

reproduce:
	$(PYTHON) scripts/generate_simulation_figures.py
verify:
	$(PYTHON) scripts/verify_results.py
test:
	$(PYTHON) -m unittest discover -s tests -v
notebook:
	$(PYTHON) scripts/run_notebook.py
smoke:
	$(PYTHON) -m functional_fitted_q smoke
plan:
	$(PYTHON) scripts/plan_experiments.py
