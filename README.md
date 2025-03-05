# Divorca - AI Divorce Predictor

## Overview
Divorca is a web application designed to predict the likelihood of divorce based on various relationship factors. Utilizing a Random Forest classifier, the application provides users with insights into their risk of divorce and the factors influencing that risk.

## Project Structure
```
Divorca
├── data
│   └── divorce_data.csv        # Dataset for predicting divorce probabilities
├── src
│   └── yu.py                   # Main application script
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd Divorca
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment. You can create one using:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   Then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   Start the Streamlit application with the following command:
   ```bash
   streamlit run src/yu.py
   ```

## Usage Guidelines
- Navigate to the web application in your browser as directed by the terminal output.
- Use the sidebar to input relationship factors and receive a prediction regarding divorce risk.
- Review the feature importance to understand which factors are most influential in the prediction.

## Ethical Considerations
This tool is intended for academic purposes only. Predicting relationship outcomes can be sensitive, and such predictions should never replace professional counseling. Bias in data can lead to unfair or inaccurate predictions.