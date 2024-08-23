# Video Game Sales Prediction

![video_Games](https://github.com/yordanossimegnew/Video_Game_Sales_Prediction/blob/main/video%20game%20sales.jpg)

![web app gif](https://github.com/yordanossimegnew/Video_Game_Sales_Prediction/blob/main/post_man.gif)

## Overview

This project involves predicting the global sales of video games based on various features. The project uses a machine learning model to forecast sales, and the pipeline is 
implemented using FastAPI to provide a web interface for predictions.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Data](#data)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Model Training](#model-training)
5. [API Interface](#api-interface)
6. [Usage](#usage)
7. [Dependencies](#dependencies)


## Project Structure

```
.
├── data
│   ├── raw data
│   │   └── video_games_sales.csv
│   └── processed data
├── models
├── notebooks.ipynb
├── preprocessor.py
├── app.py
├── config.py
├── reports
│   └── Figures
├── vgs_venv
├── requirements.txt
└── Scripts
```

- `data/raw data/video_games_sales.csv`: Contains the dataset used for training and predictions.
- `data/processed data`: Directory for storing processed data files.
- `models`: Directory for saving model files.
- `notebooks.ipynb`: Jupyter notebook for exploratory data analysis and experimentation.
- `preprocessor.py`: Contains the `Pipeline` class that handles data preprocessing and model training.
- `app.py`: Contains the FastAPI application for serving predictions.
- `config.py`: Configuration file containing paths and parameters for preprocessing.
- `reports/Figures`: Directory for storing figures and reports generated during the analysis.
- `vgs_venv`: Virtual environment directory for managing project dependencies.
- `requirements.txt`: File listing the project dependencies.
- `Scripts`: Directory for storing additional scripts related to the project.
```

## Data

The dataset used for this project is the [Video Game Sales dataset](https://www.kaggle.com/datasets/ashokselvaraj/video-game-sales-dataset). It includes various features such as:

- `Name`: Title of the game
- `Platform`: Platform of the game
- `Year_of_Release`: Release year of the game
- `Genre`: Genre of the game
- `Publisher`: Publisher of the game
- `NA_Sales`: Sales in North America
- `EU_Sales`: Sales in Europe
- `JP_Sales`: Sales in Japan
- `Other_Sales`: Sales in other regions
- `Critic_Score`: Score given by critics
- `Critic_Count`: Number of critics
- `User_Score`: Score given by users
- `User_Count`: Number of users
- `Developer`: Developer of the game
- `Rating`: Rating of the game

## Preprocessing Pipeline

The `Pipeline` class in `preprocessor.py` handles the following tasks:

1. **Data Cleaning**:
   - Drops leaky features that do not contribute to prediction.
   - Converts data types for specific features.
   - Handles high and low cardinality features.

2. **Feature Engineering**:
   - Imputes missing categorical values with the mode or a placeholder.
   - Encodes categorical variables using mean target values.
   - Imputes and processes temporal variables (e.g., release year).

3. **Scaling**:
   - Standardizes continuous variables to have zero mean and unit variance.

4. **Model Training**:
   - Uses linear regression to train the model on the preprocessed data.

## Model Training

The pipeline is trained using the `LinearRegression` model from `sklearn`. The training process involves:

1. **Splitting** the data into training and testing sets.
2. **Fitting** the preprocessing steps and the model.
3. **Evaluating** the model using metrics such as Mean Absolute Error (MAE).

## API Interface

The FastAPI application (`app.py`) provides an endpoint for making predictions:

- **GET `/`**: Returns a welcome message.
- **POST `/predict`**: Accepts a list of video game data and returns a sales prediction. The request body should be in the following format:

```json
{
  "data": [
    {
      "Name": "Game Title",
      "Platform": "Platform",
      "Year_of_Release": 2020,
      "Genre": "Genre",
      "Publisher": "Publisher",
      "NA_Sales": 1.5,
      "EU_Sales": 0.5,
      "JP_Sales": 0.2,
      "Other_Sales": 0.1,
      "Critic_Score": "80",
      "Critic_Count": 5,
      "User_Score": 8.0,
      "User_Count": 100,
      "Developer": "Developer",
      "Rating": "E"
    }
  ]
}
```

## Usage

1. **Run the FastAPI Application**:
   ```bash
   uvicorn app:app --reload
   ```
   The application will be available at `http://127.0.0.1:8000`.

2. **Make Predictions**:
   Use a tool like `curl` or Postman to send a POST request to `/predict` with the appropriate JSON payload.

## Dependencies

The project requires the following Python packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `fastapi`
- `uvicorn`
- `pydantic`

Install the dependencies using pip:

```bash
pip install numpy pandas scikit-learn fastapi uvicorn pydantic
```
