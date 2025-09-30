install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black src/

lint:
	flake8 --ignore=E203,W503,E501,E303,E305,E231,E401,F401,E302 src/ || true

clean:
	rm -rf __pycache__ .pytest_cache .coverage

all: install format 