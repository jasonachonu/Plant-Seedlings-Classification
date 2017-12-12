# Plant Seedlings Classification - Kaggle Competition

## Authors: Michael Arango, Prince Birring, & Brent Skoumal

#### This repository documents the work we did on the Plant Seedlings Classification dataset from the [kaggle competition](https://www.kaggle.com/c/plant-seedlings-classification). Our work was primarily done as a final project for a Deep Learning class. 


**Getting Started:**

1. Download the Data

The first thing you will need to do is go fetch the data from kaggle so that you can work on it either on your local machine or in the cloud. We have written a shell script that will actually go do this for you, but there are a few things you need to do before running it. 

First, you need to make sure that you have a kaggle account. Next, if your kaggle account is linked to your Google, Facebook or Yahoo account and you don't have an actual password associated with your username, you will need to set one up. The easiest way to do so is to go to the login page and click `Forgot Password`. You should receive an email from kaggle with three different options to reset your password. We want to select the third option which states: `Or if you prefer, we can set you up with your own Kaggle username/password that you can use, and connect it to your original Google account. Simply click here, and you'll be directed to a page where you will pick a username and password.` This will redirect you to a page to create your password. Then, you will need to navigate to the competition page and accept the rules of the competition. 

Assuming you have cloned this repository, you should already have the `download_data.sh` script. Change your current working directory so you are in the same directory as the script and run 
```
$ chmod +x download_data.sh
```
to make the script executable. Lastly, execute the shell script by running
```
$ ./download_data.sh
```
You will be prompted for your kaggle username and password. We have hard-coded the competition name into the script for you, but if you ever want to use this script for another challenge you just need to copy whatever comes after https://www.kaggle.com/c/ in the url to the competition page. 

This script will make a new directory `kaggle/` and download all the files from the competition into it. The path to the directory will be `~/kaggle` but feel free to change this in the script if you want it located somewhere else. 

2. Preprocessing the Data

Run the preprocessing scripts

3. Data Generator Script

Run the Generator script and change the `INPUT_SIZE` to the desired size

4. Modeling

Run the modeling scripts.

