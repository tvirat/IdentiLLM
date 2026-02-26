# IdentiLLM Dashboard

## Quick Start Guide

### Installation

Install the required dependencies:

```bash
pip install streamlit pandas numpy
```

### Running the Dashboard

To launch the dashboard, run:

```bash
streamlit run app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`
<img width="1919" height="862" alt="image" src="https://github.com/user-attachments/assets/c3a48a6e-514d-4631-a38b-3c9ef4d6b5e1" />


### Features

- **Interactive Feature Selection**: Choose values from 5 randomly selected features from your training data
- **Real-time Predictions**: Get instant predictions using your ML model
- **Clean UI**: Modern, responsive interface optimized for ML projects
- **Data Visualization**: View sample data and class distributions
- **Model Information**: Display model performance metrics

### Dashboard Components

1. **Sidebar**: Shows dataset statistics and class distribution
2. **Input Panel**: Select/enter values for features
3. **Prediction Panel**: Displays prediction results and selected features
4. **Sample Data**: Preview of actual data from your training set

### File Structure

```
CSC311_Project/
├── app.py                    # Main dashboard application
├── pred_example.py           # Prediction functions
├── training_data_clean.csv   # Training data
└── DASHBOARD_README.md       # This file
```

### Troubleshooting

**Error: Module not found**

- Make sure you've installed all dependencies: `pip install streamlit pandas numpy`

**Error: CSV file not found**

- Ensure `training_data_clean.csv` is in the same directory as `app.py`

**Port already in use**

- Use a different port: `streamlit run app.py --server.port 8502`

### Customization

You can customize the dashboard by modifying `app.py`:

- Change the number of features: Modify the `random.sample()` line
- Adjust colors: Edit the CSS in the `st.markdown()` section
- Add more visualizations: Use Streamlit's built-in charting functions

### Requirements

- Python 3.x
- streamlit
- pandas
- numpy

---

**Project**: IdentiLLM  
**Course**: CSC311 - Introduction to Machine Learning  
**Institution**: University of Toronto
