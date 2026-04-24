# Titanic Dataset Analysis

## Project Overview
This project analyzes the Titanic dataset to build a survival prediction workflow using data cleaning, feature engineering, and feature selection.

## Folder Structure
- data/
- notebooks/
- scripts/
- README.md
- requirements.txt

## Approach
1. Clean missing values, outliers, and inconsistencies
2. Engineer meaningful features such as FamilySize, IsAlone, Title, Deck, AgeGroup, and FarePerPerson
3. Encode categorical variables and transform skewed features
4. Select the most relevant features using correlation analysis and Random Forest feature importance

## Features Engineered
- FamilySize
- IsAlone
- Title
- Deck
- AgeGroup
- FarePerPerson
- FareLog
- AgeLog
- Pclass_Fare

## Data Cleaning Decisions
- Age imputed with median
- Embarked imputed with mode
- Cabin missingness handled through deck extraction and indicator columns
- Fare capped using IQR and log-transformed

## Key Findings
- Sex, Pclass, Title, Fare-related features, and family-related features were among the strongest predictors
- Log transformation reduced skewness in Fare
- Title and Deck added useful signal beyond the raw features

## How to Run
```bash
pip install -r requirements.txt
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py