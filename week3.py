# ============================================
# MONTH 5 - SENTIMENT ANALYZER
# WEEK 3 - VERSION 0.3
# Learning From Training Data
# ============================================

import json


# --------------------------------------------
# TRAINING DATA
# --------------------------------------------

training = [

    # Positive examples
    ("I love this movie", "positive"),
    ("what a great film", "positive"),
    ("this movie is amazing", "positive"),
    ("I enjoyed the story", "positive"),
    ("the food was excellent", "positive"),
    ("the service was friendly", "positive"),
    ("this product is fantastic", "positive"),
    ("I am very happy", "positive"),
    ("the experience was wonderful", "positive"),
    ("this is the best day", "positive"),
    ("the game was fun", "positive"),
    ("the teacher was helpful", "positive"),
    ("I had a great time", "positive"),
    ("the movie was brilliant", "positive"),
    ("this place is beautiful", "positive"),

    # Negative examples
    ("this movie was awful", "negative"),
    ("I hate this film", "negative"),
    ("the food was terrible", "negative"),
    ("the service was rude", "negative"),
    ("this product is broken", "negative"),
    ("I am very sad", "negative"),
    ("the experience was horrible", "negative"),
    ("this is the worst day", "negative"),
    ("the game was boring", "negative"),
    ("the teacher was unhelpful", "negative"),
    ("I am disappointed", "negative"),
    ("the movie was bad", "negative"),
    ("the service was terrible", "negative"),
    ("this place is awful", "negative"),
    ("I dislike this product", "negative")
]


# --------------------------------------------
# LEARN FROM TRAINING DATA
# --------------------------------------------

def train_model(data):

    pos_counts = {}
    neg_counts = {}

    for text, label in data:

        words = text.lower().split()

        for word in words:

            if label == "positive":

                pos_counts[word] = pos_counts.get(word, 0) + 1

            else:

                neg_counts[word] = neg_counts.get(word, 0) + 1

    model = {
        "positive": pos_counts,
        "negative": neg_counts
    }

    return model


# --------------------------------------------
# CLASSIFY NEW TEXT
# --------------------------------------------

def classify(text, model):

    words = text.lower().split()

    positive_score = 0
    negative_score = 0

    for word in words:

        positive_score += model["positive"].get(word, 0)
        negative_score += model["negative"].get(word, 0)

    if positive_score > negative_score:
        return "positive"

    elif negative_score > positive_score:
        return "negative"

    else:
        return "neutral"


# --------------------------------------------
# SAVE MODEL
# --------------------------------------------

def save_model(model, filename):

    with open(filename, "w") as file:

        json.dump(model, file, indent=4)


# --------------------------------------------
# LOAD MODEL
# --------------------------------------------

def load_model(filename):

    with open(filename, "r") as file:

        model = json.load(file)

    return model


# --------------------------------------------
# TRAIN THE AGENT
# --------------------------------------------

print("Training the sentiment analyzer...")

model = train_model(training)

print("Training complete!")

# Save learned knowledge
save_model(model, "sentiment_model.json")

print("Model saved as sentiment_model.json")


# --------------------------------------------
# TEST LEARNED MODEL
# --------------------------------------------

print()
print("Testing the learned model:")
print()

test_sentences = [
    "I love this",
    "the movie was terrible",
    "what a fantastic experience",
    "this product is awful",
    "the film was great",
    "the service was rude"
]

for sentence in test_sentences:

    result = classify(sentence, model)

    print(sentence)
    print("Prediction:", result)
    print()


# --------------------------------------------
# INTERACTIVE AGENT
# --------------------------------------------

print("===================================")
print("       SENTIMENT ANALYZER v0.3")
print("===================================")
print("The agent learned from examples.")
print("Type 'quit' to stop.")
print()


while True:

    text = input("You: ")

    if text.lower() == "quit":

        print("Agent: Goodbye!")
        break

    result = classify(text, model)

    print("Agent:", result.upper())