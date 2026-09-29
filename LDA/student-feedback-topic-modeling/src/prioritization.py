def calculate_priority(topic_data):
    """
    topic_data: dict with keys:
    - frequency: int
    - avg_strength: float
    - avg_neg_sentiment: float
    """
    freq = topic_data['frequency']
    strength = topic_data['avg_strength']
    neg = topic_data['avg_neg_sentiment']
    
    # Priority score formula
    score = freq * strength * (neg + 0.1)
    
    # Categorize based on standard distribution for 150-record dataset
    if score >= 7.0:
        level = "High"
    elif score >= 3.0:
        level = "Medium"
    else:
        level = "Low"
        
    return score, level
