# Machine Learning - predicting the Life Expectancy in a Country
This project returns 4 machine learning models that predict, given some data, the life expectancy in a country. Two of the models use Linear Regression (one with and another without cross validation), while the other two make use of Random Forest (again - one with and another without cross validation).


## How to run

### Requirements
There are two ways to generate the ml models: the first one is by using the jupyter notebook file (main.ipynb), while the second only requires a Python installation.

- Python 3
- Jupyter Notebook

### Step-by-step - common steps
- Clone the project
- On the root dir, create virtual environment (python3 -m venv .venv)
- On the same dir, start the virtual environment (source .venv/bin/activate)
- Install packages listed on requirements.txt (pip3 install -r requirements.txt)

#### Step-by-step - jupyter notebook file
- Create a jupyter kernel (ipython kernel install --user --name=.venv)
- Start jupyter notebook, open the main.ipynb file, select the .venv kernel and then run all the cels.
- Make sure you delete the kernel once you are done (jupyter-kernelspec uninstall ..venv).

#### Step-by-step - python file
- On the root dir, enter python3 'main.py'.