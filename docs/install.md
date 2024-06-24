# Installation

## Running the tutorial notebooks

You can run the tutorial notebooks directly in Google Colab. To correctly install all required packages, make sure to run the command

``` py
!pip install git+https://github.com/binado/sbisandbox.git
```

in an empty cell in the beginning of the notebook.

## Local installation

Clone the repo with

``` bash
git clone https://github.com/binado/sbisandbox.git
```

and run

``` bash
pip install .
```

If possible, we recommend creating a new `conda` environment first:

``` bash
conda create -n sbisandbox python=3.10 && conda activate sbisandbox
```
