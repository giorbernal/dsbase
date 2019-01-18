#!/bin/bash

touch dsbase.log

IS_DSBASE=$(/opt/conda/bin/conda-env list | grep dsbase | wc -l)

# Installing/charging environment
if [ $IS_DSBASE == 0 ]; then
	echo "creating dsbae environment ..." >> dsbase.log
	/opt/conda/bin/conda create -n dsbase python=3.6.3 -y
	echo "dsbae environment created!" >> dsbase.log
	source activate dsbase
	echo "inside dsbase environment!. Intalling the whole stuff ..." >> dsbase.log
	/opt/conda/bin/conda install numpy pandas scikit-learn matplotlib seaborn -y && \
	echo "   ... numpy pandas scikit-learn matplotlib seaborn installed!" >> dsbase.log
	/opt/conda/bin/conda install tensorflow -y && \
	echo "   ... tensorflow installed!" >> dsbase.log
	/opt/conda/bin/conda install keras -y && \
	echo "   ... keras installed!" >> dsbase.log
	/opt/conda/bin/conda install jupyter -y
	echo "   ... jupyter installed!" >> dsbase.log
	pip install kaggle
	echo "   ... kaggle installed!" >> dsbase.log

	echo "Everything installed!!" >> dsbase.log
else
	echo "dsbase environment previuosly installed!" >> dsbase.log
	source activate dsbase
	echo "inside dsbase environment!" >> dsbase.log
fi

# Setting kaggle file
KAGGLE_FILE=/root/.kaggle/kaggle.json
sed -i s/KAGGLE_USER/$KAGGLE_USER/g $KAGGLE_FILE
sed -i s/KAGGLE_KEY/$KAGGLE_KEY/g $KAGGLE_FILE
echo "kaggle.json setted!" >> dsbase.log

echo "Initiate Jupyter Notebook!" >> dsbase.log
/run_jupyter.sh