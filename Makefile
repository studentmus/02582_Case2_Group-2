.PHONY: setup test lint clean

setup:
	 bash setup.sh

test:
	 pytest -q

clean:
	 find . -type d -name "__pycache__" -exec rm -rf {} +
	 find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
