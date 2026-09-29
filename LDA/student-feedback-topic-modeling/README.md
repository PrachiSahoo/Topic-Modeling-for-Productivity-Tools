# Student Feedback Topic Modeling

An NLP-powered dashboard for analyzing student app feedback.

## Features
- **Topic Modeling**: Uses LDA to discover underlying themes.
- **Sentiment Analysis**: Uses VADER to calculate negativity scores.
- **Prioritization**: Automatically scores topics based on frequency and negative sentiment.
- **Dashboard**: A Streamlit web interface to explore data.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python train.py`
3. Run the dashboard: `streamlit run app.py`
