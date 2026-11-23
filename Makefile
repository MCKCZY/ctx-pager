PYTHON ?= python3

.PHONY: test bench demo check clean

check:
	$(PYTHON) -m compileall -q src benchmarks examples tests skills/ctx-pager/scripts scripts
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
	PYTHONPATH=src $(PYTHON) examples/demo_small_context_session.py > /dev/null

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

bench:
	PYTHONPATH=src $(PYTHON) -m benchmarks.run_benchmark

demo:
	PYTHONPATH=src $(PYTHON) examples/demo_small_context_session.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info .ctx_pager
