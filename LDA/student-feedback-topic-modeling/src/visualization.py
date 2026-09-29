import plotly.express as px
import pandas as pd

def plot_topic_distribution(df):
    counts = df['Dominant_Topic'].value_counts().reset_index()
    counts.columns = ['Topic', 'Count']
    counts['Topic'] = counts['Topic'].apply(lambda x: f"Topic {x}")
    fig = px.pie(counts, values='Count', names='Topic', title='Topic Distribution', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(
        template="plotly_dark",
        font=dict(size=16, family="Outfit, sans-serif", color="#f8fafc"),
        title_font_size=24,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    return fig

def plot_priority_scatter(topics_summary):
    df = pd.DataFrame(topics_summary)
    if len(df) == 0:
        return px.scatter(title="No topics found")
        
    df['Topic Name'] = df['topic_id'].apply(lambda x: f"Topic {x}")
    fig = px.scatter(df, x='frequency', y='avg_neg_sentiment', size='priority_score', 
                     color='priority_level', hover_data=['keywords'],
                     title='Priority Matrix: Frequency vs Sentiment',
                     color_discrete_map={'High': '#ef4444', 'Medium': '#f97316', 'Low': '#3b82f6'},
                     size_max=55)
    fig.update_layout(
        template="plotly_dark",
        font=dict(size=14, family="Outfit, sans-serif", color="#f8fafc"),
        title_font_size=22,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    return fig

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_wordcloud(keywords, topic_id):
    wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA', colormap='plasma').generate(" ".join(keywords))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.set_title(f"Topic {topic_id} Keywords", fontsize=24, color='#f8fafc', fontfamily='sans-serif', pad=20)
    return fig
