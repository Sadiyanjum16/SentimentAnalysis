# ============================================
# MONTH 5 - SENTIMENT ANALYZER
# WEEK 2 - VERSION 0.2
# Probability, Confidence & Negation
# ============================================


# --------------------------------------------
# WORD LISTS
# --------------------------------------------

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
# ANALYZE FUNCTION
# --------------------------------------------

def analyze(text):

    text = text.lower()

    words = text.split()

    pos = 0
    neg = 0

    # Check every word
    for i, word in enumerate(words):

        # Check for "not" before a positive word
        if word in positive_words:

            if i > 0 and words[i - 1] == "not":
                neg += 1
            else:
                pos += 1

        # Check for "not" before a negative word
        elif word in negative_words:

            if i > 0 and words[i - 1] == "not":
                pos += 1
            else:
                neg += 1

    # Total mood words
    total = pos + neg

    # No mood words found
    if total == 0:
        return "neutral", 0.0, pos, neg

    # Positive sentiment
    if pos > neg:
        confidence = pos / total
        return "positive", confidence, pos, neg

    # Negative sentiment
    elif neg > pos:
        confidence = neg / total
        return "negative", confidence, pos, neg

    # Equal counts
    else:
        return "neutral", 0.5, pos, neg


# --------------------------------------------
# CONFIDENCE BAR
# --------------------------------------------

def confidence_bar(confidence):

    percentage = int(confidence * 100)

    bars = int(confidence * 10)

    return "[" + "#" * bars + "-" * (10 - bars) + "]"


# --------------------------------------------
# INTERACTIVE AGENT
# --------------------------------------------

print("===================================")
print("       SENTIMENT ANALYZER v0.2")
print("===================================")
print("I can detect sentiment and confidence.")
print("I also understand simple 'not' phrases.")
print("Type 'quit' to stop.")
print()


while True:

    text = input("You: ")

    if text.lower() == "quit":
        print("Agent: Goodbye!")
        break

    label, confidence, pos, neg = analyze(text)

    percentage = round(confidence * 100)

    print()
    print("Agent: Sentiment =", label.upper())
    print("Agent: Confidence =", percentage, "%")
    print("Agent:", confidence_bar(confidence))
    print("Agent: Positive evidence =", pos)
    print("Agent: Negative evidence =", neg)
    print()