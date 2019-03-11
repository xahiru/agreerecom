# AgreeTrust

## requirement

python version = 3.7.1
pandas
surprise from 'https://github.com/NicolasHug/Surprise'

clone this project

The processed weight matrixes can be saved for later use in the location set by
file_path_save_data variable in 'agreetrust.py'
The respective folders must exist (you need to manually create them)
Currently it is set to 'data/processed/'

## compile 
From the project root folder run:

``cd surprise; python setup.py install;`` 

## run
To execute the prediction algrothm evaluation, from the project root folder run:

``python agreetrust.py``

For time and sparsity evaluation run:

``python timensparsityeval.py``

## Algorithms
AgreeTrust and O'dnovan algorithms are implemented in the 'agreements.pyx' file in 'surprise/surprise/prediction_algorithms' folder
