from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib

class TopicModelPipeline:
    def __init__(self, n_topics=5, n_words=5):
        self.n_topics = n_topics
        self.n_words = n_words
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        
    def fit(self, texts):
        dtm = self.vectorizer.fit_transform(texts)
        self.lda.fit(dtm)
        
    def transform(self, texts):
        dtm = self.vectorizer.transform(texts)
        return self.lda.transform(dtm)
        
    def get_topics(self):
        words = self.vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [words[i] for i in topic.argsort()[:-self.n_words - 1:-1]]
            topics[topic_idx] = top_words
        return topics

    def save(self, path):
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path):
        return joblib.load(path)
