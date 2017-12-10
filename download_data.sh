#! /bin/bash

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python-pip
sudo pip install kaggle-cli
sudo pip install -U kaggle-cli
read -p 'Competition:' competition
read -p 'Username:' uservar
read -sp 'Password:' passvar
kg config -u $uservar -p $passvar -c $competition
kg download
sudo apt-get install unzip
unzip train.zip
unzip test.zip
