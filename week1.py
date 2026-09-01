# ============================================
# MONTH 5 - SENTIMENT ANALYZER
# WEEK 1 - VERSION 0.1
# Words as Signals & Counting
# ============================================

# Positive words the agent knows
positive_words = [
    "good",
    "great",
    "love",
    "happy",
    "excellent",
    "best",
    "amazing",
    "awesome",
    "wonderful",
    "fantastic",
    "nice",
    "beautiful",
    "enjoy",
    "enjoyed",
    "perfect",
    "fun",
    "helpful",
    "friendly",
    "brilliant",
    "super"
]

# Negative words the agent knows
negative_words = [
    "bad",
    "hate",
    "awful",
    "terrible",
    "worst",
    "sad",
    "horrible",
    "poor",
    "boring",
    "angry",
    "disappointed",
    "annoying",
    "rude",
    "useless",
    "broken",
    "slow",
    "difficult",
    "unhappy",
    "dislike"
]


# --------------------------------------------
# SENTIMENT ANALYZER FUNCTION
# --------------------------------------------

def analyze(text):

    # Convert everything to lowercase
    text = text.lower()

    # Split sentence into individual words
    words = text.split()

    # Count positive and negative words
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)

    # Decide sentiment
    if pos_count > neg_count:
        sentiment = "positive"

    elif neg_count > pos_count:
        sentiment = "negative"

    else:
        sentiment = "neutral"

    return sentiment


# --------------------------------------------
# INTERACTIVE AGENT
# --------------------------------------------

print("===================================")
print("       SENTIMENT ANALYZER v0.1")
print("===================================")
print("Type a sentence and I will detect")
print("its mood.")
print("Type 'quit' to stop.")
print()


while True:

    text = input("You: ")

    # Stop the program
    if text.lower() == "quit":
        print("Agent: Goodbye!")
        break

    # Analyze the sentence
    sentiment = analyze(text)

    # Display result
    if sentiment == "positive":
        print("Agent: Sentiment = POSITIVE 🙂")

    elif sentiment == "negative":
        print("Agent: Sentiment = NEGATIVE 🙁")

    else:
        print("Agent: Sentiment = NEUTRAL 😐")