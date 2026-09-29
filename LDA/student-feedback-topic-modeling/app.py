import streamlit as st
import pandas as pd
import os
from src.preprocessing import clean_text
from src.sentiment import get_sentiment
from src.topic_model import TopicModelPipeline
from src.prioritization import calculate_priority
from src.visualization import plot_topic_distribution, plot_priority_scatter, plot_wordcloud

st.set_page_config(page_title="Feedback Analytics", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for premium vibrant aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Base font increase */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 18px !important;
        color: #f8fafc !important;
    }
    
    /* Dark background */
    .stApp {
        background-color: #0f172a;
    }

    /* Title styling */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #818cf8, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px !important;
        line-height: 1.2 !important;
    }
    h2 { font-size: 2.2rem !important; font-weight: 700 !important; color: #f1f5f9 !important; }
    h3 { font-size: 1.6rem !important; font-weight: 600 !important; color: #e2e8f0 !important; }

    /* Glassmorphism for Metric Cards in Dark Mode */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
    }
    div[data-testid="stMetricValue"] {
        font-size: 4rem !important;
        font-weight: 800 !important;
        color: #818cf8 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
    }

    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        padding: 10px 30px;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #94a3b8 !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(30, 41, 59, 1) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        color: #818cf8 !important;
        border-bottom-color: transparent !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #f1f5f9 !important;
    }
    .streamlit-expanderContent {
        font-size: 1.2rem !important;
        color: #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Student Feedback Analytics")

# Check if model exists
model_path = 'models/lda_model.pkl'
if not os.path.exists(model_path):
    st.warning(f"Model not found at '{model_path}'. Please run `python train.py` first to train the model.")
    st.stop()

@st.cache_resource
def load_model():
    return TopicModelPipeline.load(model_path)

pipeline = load_model()
topics_keywords = pipeline.get_topics()

st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Feedback CSV", type="csv")

# Require a file upload to proceed
if uploaded_file is None:
    st.info("Please upload a CSV file in the sidebar to get started.")
    st.stop()
else:
    df = pd.read_csv(uploaded_file)

st.sidebar.markdown("---")
st.sidebar.header("Manual Inference")
manual_text = st.sidebar.text_area("Enter feedback manually:")
if st.sidebar.button("Analyze Manual Feedback"):
    if manual_text.strip():
        ct = clean_text(manual_text)
        dist = pipeline.transform([ct])[0]
        dom_topic = dist.argmax()
        s = get_sentiment(manual_text)
        st.sidebar.success(f"**Topic**: {dom_topic}\n\n**Sentiment**: {s[0]:.2f}")
    else:
        st.sidebar.warning("Please enter some text.")

if 'text' not in df.columns:
    st.error("CSV must contain a 'text' column.")
else:
    with st.spinner("Analyzing feedback..."):
        df['clean_text'] = df['text'].apply(clean_text)
        
        # Sentiment
        sentiments = df['text'].apply(get_sentiment)
        df['compound_sentiment'] = [s[0] for s in sentiments]
        df['negative_sentiment'] = [s[1] for s in sentiments]
        
        # Topic Modeling
        topic_dists = pipeline.transform(df['clean_text'])
        df['Dominant_Topic'] = topic_dists.argmax(axis=1)
        df['Topic_Strength'] = topic_dists.max(axis=1)
        
        # Aggregate Topics
        topics_summary = []
        for topic_id in range(pipeline.n_topics):
            topic_df = df[df['Dominant_Topic'] == topic_id]
            freq = len(topic_df)
            if freq > 0:
                avg_strength = topic_df['Topic_Strength'].mean()
                avg_neg = topic_df['negative_sentiment'].mean()
                
                priority_score, priority_level = calculate_priority({
                    'frequency': freq,
                    'avg_strength': avg_strength,
                    'avg_neg_sentiment': avg_neg
                })
                
                topics_summary.append({
                    'topic_id': topic_id,
                    'keywords': ", ".join(topics_keywords.get(topic_id, [])),
                    'frequency': freq,
                    'avg_strength': avg_strength,
                    'avg_neg_sentiment': avg_neg,
                    'priority_score': priority_score,
                    'priority_level': priority_level
                })
        
        # Sort by priority score
        topics_summary.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Dynamically assign priority levels based on rank to ensure even distribution
        n = len(topics_summary)
        for i, t in enumerate(topics_summary):
            if i < n // 3 or (i == 0 and n < 3):
                t['priority_level'] = 'High'
            elif i < 2 * (n / 3.0):
                t['priority_level'] = 'Medium'
            else:
                t['priority_level'] = 'Low'
        
        # KPI Metrics
        st.markdown("---")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Total Feedback", len(df))
        with kpi2:
            st.metric("Topics Discovered", pipeline.n_topics)
        with kpi3:
            high_priority_count = len([x for x in topics_summary if x['priority_level'] == 'High'])
            st.metric("High Priority Issues", high_priority_count)
        
        st.markdown("---")
        
        # Tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📊 Overview & Roadmap", "☁️ Topic Word Clouds", "🔎 Feedback Explorer"])
        
        with tab1:
            # Visualizations
            st.plotly_chart(plot_priority_scatter(topics_summary))
            st.markdown("<br><hr><br>", unsafe_allow_html=True)
            st.plotly_chart(plot_topic_distribution(df))
                
            # Roadmap / Priorities
            st.header("🎯 Feature Roadmap Priorities")
            
            high_col, med_col, low_col = st.columns(3)
            with high_col:
                st.subheader("🔴 High Priority")
                for t in [x for x in topics_summary if x['priority_level'] == 'High']:
                    with st.expander(f"Topic {t['topic_id']} ({t['frequency']} feedback)"):
                        st.write(f"**Keywords**: {t['keywords']}")
                        st.write(f"**Score**: {t['priority_score']:.2f}")

            with med_col:
                st.subheader("🟠 Medium Priority")
                for t in [x for x in topics_summary if x['priority_level'] == 'Medium']:
                    with st.expander(f"Topic {t['topic_id']} ({t['frequency']} feedback)"):
                        st.write(f"**Keywords**: {t['keywords']}")
                        st.write(f"**Score**: {t['priority_score']:.2f}")
                        
            with low_col:
                st.subheader("🔵 Low Priority")
                for t in [x for x in topics_summary if x['priority_level'] == 'Low']:
                    with st.expander(f"Topic {t['topic_id']} ({t['frequency']} feedback)"):
                        st.write(f"**Keywords**: {t['keywords']}")
                        st.write(f"**Score**: {t['priority_score']:.2f}")
                        
        with tab2:
            st.header("☁️ Topic Word Clouds")
            wc_cols = st.columns(3)
            for topic_id in range(pipeline.n_topics):
                kw = topics_keywords.get(topic_id, [])
                if kw:
                    fig = plot_wordcloud(kw, topic_id)
                    wc_cols[topic_id % 3].pyplot(fig)
            
        with tab3:
            st.header("🔎 Feedback Explorer")
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                topic_filter = st.selectbox("Filter by Topic", ["All"] + [f"Topic {i}" for i in range(pipeline.n_topics)])
            with filter_col2:
                sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Negative", "Positive", "Neutral"])
            
            filtered_df = df.copy()
            if topic_filter != "All":
                tid = int(topic_filter.split(" ")[1])
                filtered_df = filtered_df[filtered_df['Dominant_Topic'] == tid]
            
            if sentiment_filter == "Negative":
                filtered_df = filtered_df[filtered_df['compound_sentiment'] < -0.05]
            elif sentiment_filter == "Positive":
                filtered_df = filtered_df[filtered_df['compound_sentiment'] > 0.05]
            elif sentiment_filter == "Neutral":
                filtered_df = filtered_df[(filtered_df['compound_sentiment'] >= -0.05) & (filtered_df['compound_sentiment'] <= 0.05)]
                
            st.dataframe(filtered_df[['text', 'Dominant_Topic', 'compound_sentiment', 'negative_sentiment']])

