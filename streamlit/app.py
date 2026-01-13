"""
IdentiLLM Dashboard - Interactive ML Prediction Interface
A Streamlit-based UI for the IdentiLLM project
"""

import streamlit as st
import pandas as pd
from pred_example import predict
from questions import LIKELIHOOD_OPTIONS, LIKERT_OPTIONS, QUESTIONS

# Page configuration
st.set_page_config(
    page_title="IdentiLLM Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #145a8a;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .question-number {
        display: inline-block;
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 50%;
        font-weight: bold;
        margin-right: 0.5rem;
        font-size: 0.9rem;
    }
    .mandatory-badge {
        display: inline-block;
        background-color: #dc3545;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-left: 0.5rem;
        font-weight: bold;
    }
    .optional-badge {
        display: inline-block;
        background-color: #6c757d;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-left: 0.5rem;
        font-weight: normal;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('training_data_clean.csv')
    return df

# Initialize session state
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">🤖 IdentiLLM Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict which LLM a student is describing: ChatGPT, Claude, or Gemini</div>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ Error: 'training_data_clean.csv' not found. Please ensure the file is in the same directory.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dataset Info")
        st.metric("Total Samples", len(df))
        st.metric("Features", len(df.columns) - 1)  # Excluding label
        
        if 'label' in df.columns:
            st.markdown("### Class Distribution")
            label_counts = df['label'].value_counts()
            for label, count in label_counts.items():
                st.write(f"**{label}:** {count}")
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This dashboard allows you to input feature values and predict which LLM (ChatGPT, Claude, or Gemini) is being described.")
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("📝 Survey Questions")
        st.markdown("Please answer the following questions about the LLM you are evaluating:")
        
        # Display all questions
        for question in QUESTIONS:
            with st.container():
                # Question header with number and mandatory badge
                mandatory_badge = '<span class="mandatory-badge">REQUIRED</span>' if question['mandatory'] else '<span class="optional-badge">Optional</span>'
                st.markdown(
                    f'<div><span class="question-number">Q{question["number"]}</span>'
                    f'<strong>{question["text"]}</strong> {mandatory_badge}</div>',
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)
                
                question_key = question['key']
                
                # Open-ended text question
                if question['type'] == 'open_ended':
                    default_text = st.session_state.selected_features.get(question_key, "")
                    selected_value = st.text_area(
                        f"Your response",
                        value=default_text,
                        placeholder="Type your answer here...",
                        key=f"q_{question['number']}",
                        label_visibility="collapsed",
                        height=10
                    )
                    if selected_value:
                        st.session_state.selected_features[question_key] = selected_value
                    elif question_key in st.session_state.selected_features:
                        del st.session_state.selected_features[question_key]
                
                # Likert scale (likelihood)
                elif question['type'] == 'likert_likelihood':
                    default_value = st.session_state.selected_features.get(question_key, "")
                    options_list = [""] + LIKELIHOOD_OPTIONS
                    default_index = 0
                    if default_value in options_list:
                        default_index = options_list.index(default_value)
                    
                    selected_value = st.selectbox(
                        "Select your response",
                        options=options_list,
                        index=default_index,
                        key=f"q_{question['number']}",
                        label_visibility="collapsed"
                    )
                    if selected_value:
                        st.session_state.selected_features[question_key] = selected_value
                    elif question_key in st.session_state.selected_features:
                        del st.session_state.selected_features[question_key]
                
                # Likert scale (frequency)
                elif question['type'] == 'likert':
                    default_value = st.session_state.selected_features.get(question_key, "")
                    options_list = [""] + LIKERT_OPTIONS
                    default_index = 0
                    if default_value in options_list:
                        default_index = options_list.index(default_value)
                    
                    selected_value = st.selectbox(
                        "Select your response",
                        options=options_list,
                        index=default_index,
                        key=f"q_{question['number']}",
                        label_visibility="collapsed"
                    )
                    if selected_value:
                        st.session_state.selected_features[question_key] = selected_value
                    elif question_key in st.session_state.selected_features:
                        del st.session_state.selected_features[question_key]
                
                # Multiple choice (select all that apply)
                elif question['type'] == 'multiple_choice':
                    default_values = st.session_state.selected_features.get(question_key, [])
                    if isinstance(default_values, str):
                        # Convert comma-separated string to list
                        default_values = [v.strip() for v in default_values.split(',') if v.strip()]
                    
                    # Filter default values to only include those that exist in options
                    default_values = [v for v in default_values if v in question['options']]
                    
                    selected_values = st.multiselect(
                        "Select all that apply",
                        options=question['options'],
                        default=default_values,
                        key=f"q_{question['number']}",
                        label_visibility="collapsed"
                    )
                    if selected_values:
                        # Store as comma-separated string to match CSV format
                        st.session_state.selected_features[question_key] = ','.join(selected_values)
                    elif question_key in st.session_state.selected_features:
                        del st.session_state.selected_features[question_key]
        
        # Prediction button
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button("🔮 Make Prediction", use_container_width=True)
        
        if predict_button:
            # Check if all mandatory questions are answered
            missing_required = []
            for question in QUESTIONS:
                if question['mandatory']:
                    value = st.session_state.selected_features.get(question['key'], "")
                    # Check if value exists and is not empty
                    if not value or (isinstance(value, str) and value.strip() == ""):
                        missing_required.append(f"Q{question['number']}")
            
            if missing_required:
                st.error(f"⚠️ Please answer all required questions: {', '.join(missing_required)}")
            elif len(st.session_state.selected_features) > 0:
                # Create a row with selected features
                row_data = {col: '' for col in df.columns if col != 'label'}
                row_data.update(st.session_state.selected_features)
                row = pd.Series(row_data)
                
                # Make prediction
                prediction = predict(row)
                st.session_state.prediction_result = prediction
                st.success("✅ Prediction completed!")
            else:
                st.warning("⚠️ Please select at least one feature value before making a prediction.")
    
    with col2:
        st.header("🎯 Prediction Result")
        
        if st.session_state.prediction_result:
            # Display prediction with styling
            st.markdown(f'<div class="prediction-box">Predicted LLM: {st.session_state.prediction_result}</div>', 
                       unsafe_allow_html=True)
            
            # Display selected features
            st.markdown("### Selected Features:")
            if st.session_state.selected_features:
                for feature, value in st.session_state.selected_features.items():
                    st.markdown(f"**{feature}:**")
                    st.text(str(value)[:100] + "..." if len(str(value)) > 100 else str(value))
            
            # Model confidence visualization (placeholder)
            st.markdown("---")
            st.markdown("### Model Information")
            st.info("""
            **Model Type:** Neural Network  
            **Test Accuracy:** 67.4%  
            **Classes:** ChatGPT, Claude, Gemini
            """)
            
        else:
            st.info("👆 Select feature values and click 'Make Prediction' to see the result.")
            
            # Sample data preview
            st.markdown("### Sample Data Preview")
            preview_cols = [q['key'] for q in QUESTIONS[:3]]
            if 'label' in df.columns:
                preview_cols.append('label')
            sample_df = df.sample(min(3, len(df)))[preview_cols]
            st.dataframe(sample_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; padding: 1rem;'>"
        "IdentiLLM Dashboard | CSC311 Machine Learning Project | University of Toronto"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
