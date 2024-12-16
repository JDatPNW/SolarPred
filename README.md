# Solar Power Prediction App
## Chonnam National University Captstone Project
### Data obtained from:
- [Weather Data](https://data.kma.go.kr/)
- [Geenration Data](https://www.data.go.kr/data/15099650/fileData.do)
### Web Interface Based on:
- [Soft UI Dashboard 3](https://github.com/creativetimofficial/soft-ui-dashboard)
### Deep Learning Model:
- LSTM (based on [Approach found here](https://www.kaggle.com/code/energyenthusiast/solar-forecasting-lstm))
### Versions:
- MySQL Workbench 8.0.40
- Python 3.11.5
- tensorflow 2.15.0
- pandas 2.2.2
- numpy 1.26.2
- matplotlib 3.8.4
- seaborn 0.13.2
- scikit-learn 1.4.2
- flask 3.0.3
- sqlalchemy 2.0.36
### Structure
- `./Model Training`: dataset as .csv and Jupyter Notebook file containing:
  - Data Analysis
  - Model Training
  - Model Comparison
  - Model Evaluation
- `./Weblication`: modified version of the dashboard template to our needs running on Flask. Contains:
  - Saved LSTM model for predicting
  - Flask run file
  - HTML templates
- `Database Dump`: a dumped version of the MySQL Database as a .sql file.
### Usage
- Create MySQL server from file & run
- Enter your database URI in run.py
- Run the run.py file in the Weblication folder to start the Flask Sever
- Load website
