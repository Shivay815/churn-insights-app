.PHONY: setup data features train test all

PLATFORM_DIR := ../ecom-analytics-platform

setup:            ## install python deps
	pip install -r requirements.txt

data:             ## build Project A's warehouse (source of all features)
	@if [ ! -d $(PLATFORM_DIR) ]; then \
		git clone --depth 1 https://github.com/Shivay815/ecom-analytics-platform $(PLATFORM_DIR); \
	fi
	cd $(PLATFORM_DIR) && pip install -r requirements.txt && $(MAKE) load build

features:         ## build training table from the warehouse
	python features/build_features.py

train:            ## train + evaluate + quality gate + write artifacts
	python training/train.py

test:             ## run pytest suite
	pytest tests/ -q

all: data features train test
