.PHONY: install test run ui clean

install:
	pip install -r requirements.txt

run:
	mkdir -p outputs
	python -m app.cli --input data/eval_dataset.csv --output outputs/eval_results.csv

ui:
	streamlit run app/ui.py --server.address 0.0.0.0 --server.port 8501

test:
	pytest -q

clean:
	rm -rf outputs .pytest_cache __pycache__ app/__pycache__ tests/__pycache__
