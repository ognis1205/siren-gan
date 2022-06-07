.PHONY: clean data lint requirements install help
.DEFAULT_GOAL := help
ROOT := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

## Install Python Dependencies
requirements:
	@pip install -U pip setuptools wheel
	@pip install -r requirements.txt

## Train Model
dcgan: data
	@echo "Training GAN model"
	@train dcgan $(ROOT)/data/raw/cats $(ROOT)/reports/figures/dcgan $(ROOT)/models/dcgan
	@echo "Trained  GAN model: "$(ROOT)/models/dcgan

## Make Dataset
data: install
	@echo "Downloading cat image dataset"
	@download 1KTF-OLTxijRwPbcNJdNMHYcZtIqPKksp $(ROOT)/data/raw
	@echo "Downloaded cat image dataset: "$(ROOT)/data/raw/cats

## Delete all compiled Python files
clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	@flake8 src

## Install the package
install: requirements
	@pip install .

## Display command usage
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
