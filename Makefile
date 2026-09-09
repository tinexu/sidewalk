# Stage 0 task runner.
export PYTHONPATH := src

.PHONY: help install test demo plan survey clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install pinned dependencies
	pip install -r requirements.txt

test:  ## Run the test suite (no network, no token)
	python3 -m pytest tests/ -q

demo:  ## End-to-end demo against a synthetic API (no token)
	python3 scripts/demo_offline.py

plan:  ## Tile and cost estimate (no network)
	python3 -m reachable.cli plan

survey:  ## THE DAY-1 GATE. Needs MAPILLARY_TOKEN.
	python3 -m reachable.cli survey --out reports/

clean:
	rm -rf data/cache reports .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true