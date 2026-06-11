.PHONY: install install-all data teacher train eval test lint clean all

CONFIG ?= configs/config.yaml

install:        ## Core deps only (data + eval pipeline, no GPU)
	pip install -e ".[dev]"

install-all:    ## Everything (teacher APIs + GPU training)
	pip install -e ".[teacher,train,dev]"

data:           ## 1. Download + sample ESCI
	python scripts/01_prepare_data.py --config $(CONFIG)

teacher:        ## 2. Generate teacher labels + rationales
	python scripts/02_generate_teacher.py --config $(CONFIG)

train:          ## 3. QLoRA fine-tune the student
	python scripts/03_train_student.py --config $(CONFIG)

eval:           ## 4. Score student vs gold labels
	python scripts/04_evaluate.py --config $(CONFIG)

all: data teacher train eval  ## Run the full pipeline

test:
	pytest -q

lint:
	ruff check src scripts tests

clean:          ## Remove generated data + artifacts
	rm -rf data/interim/* data/processed/* artifacts/*
