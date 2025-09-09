install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black src/

clean:
	rm -rf __pycache__ .pytest_cache .coverage

all: install format 