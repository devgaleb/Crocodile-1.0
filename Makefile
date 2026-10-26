.PHONY: format

format:
	isort generate.py Crocodile
	yapf -i -r *.py generate.py Crocodile
