#!/bin/bash
sed -i 's/matplotlib widget/matplotlib inline/g' *.ipynb
jupyter nbconvert --to notebook --inplace --execute *.ipynb
#jupyter-nbconvert --to PDF --no-input *.ipynb
jupyter-nbconvert --to PDF *.ipynb
sed -i 's/matplotlib inline/matplotlib widget/g' *.ipynb
