import pandas as pd
import random

categories = {
    "Pricing": [
        "The subscription is too expensive for students.",
        "Can you add a family plan?",
        "I don't understand why basic features are paywalled.",
        "Give us a discount for annual billing.",
        "The free tier is way too restrictive.",
        "I would pay if it was a one-time purchase.",
        "Stop pushing the premium upgrade on every screen.",
        "I love the app but can't justify the monthly cost.",
        "The pricing model is very confusing.",
        "Add a student discount please!"
    ],
    "Offline Mode": [
        "I need to study on the train without wifi.",
        "The app is useless without an internet connection.",
        "Please let me download my notes for offline access.",
        "Offline syncing is broken.",
        "I lost all my edits when my connection dropped.",
        "Add a proper offline mode.",
        "It takes forever to reconnect to the server.",
        "Why does it need internet to view already opened documents?",
        "Please add a button to manually trigger a sync.",
        "Offline caching is taking up too much storage."
    ],
    "Performance": [
        "The app is incredibly slow on my older phone.",
        "It takes 10 seconds just to load the dashboard.",
        "Battery drains super fast when using this app.",
        "My phone overheats when I leave the app open.",
        "The scrolling is very laggy.",
        "Typing lag is unbearable, please fix.",
        "It crashes every time I try to upload an image.",
        "Please optimize the memory usage.",
        "Animations are choppy.",
        "Takes too long to start up."
    ],
    "Content Quality": [
        "The practice questions are too easy.",
        "I found several typos in the biology section.",
        "Please add more advanced math topics.",
        "Some of the video tutorials are outdated.",
        "The explanations for wrong answers are not helpful.",
        "I need more examples for programming concepts.",
        "The flashcards are great but need more detail.",
        "Content seems heavily biased towards US curriculum.",
        "Add more interactive quizzes instead of just text.",
        "The history module is missing key events."
    ],
    "Customer Support": [
        "I submitted a ticket 3 days ago and got no response.",
        "The AI chatbot is useless, let me talk to a human.",
        "Support team was incredibly helpful and fast.",
        "I can't find the FAQ section.",
        "Please add a live chat feature.",
        "The refund process was a nightmare.",
        "Where do I report bugs?",
        "I never got my password reset email.",
        "Support articles are outdated.",
        "Great customer service experience."
    ]
}

data = []
for i in range(1000):
    cat = random.choice(list(categories.keys()))
    text = random.choice(categories[cat])
    
    # Add random noise/variations
    if random.random() > 0.6:
        text = text.lower()
    if random.random() > 0.8:
        text = text + [" Please!", " Fix this asap.", " Thanks.", " ASAP", " 😡", " 👍"][random.randint(0,5)]
        
    date = f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    data.append([f"F{i:04d}", text, date])

df = pd.DataFrame(data, columns=['feedback_id', 'text', 'date'])
df.to_csv('data/student_feedback.csv', index=False)
print("Generated 1000 rows of student feedback.")
