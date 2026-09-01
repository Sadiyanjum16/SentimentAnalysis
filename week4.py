# ============================================
# MONTH 5 - SENTIMENT ANALYZER
# WEEK 4 - VERSION 1.0
# Accuracy, Testing & Final Agent
# ============================================

import json


# --------------------------------------------
# COMPLETE LABELLED DATASET
# --------------------------------------------

all_data = [

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
# SPLIT DATA
# --------------------------------------------

split = int(len(all_data) * 0.8)

train_set = all_data[:split]

test_set = all_data[split:]


print("Training on:", len(train_set))
print("Testing on:", len(test_set))


# --------------------------------------------
# TRAIN MODEL
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

    return {
        "positive": pos_counts,
        "negative": neg_counts
    }


# --------------------------------------------
# CLASSIFIER
# --------------------------------------------

def classify(text, model):

    text = text.lower().strip()

    # Empty input
    if not text:

        return "neutral", 0.0

    words = text.split()

    positive_score = 0
    negative_score = 0

    known_words = 0

    for word in words:

        pos_value = model["positive"].get(word, 0)
        neg_value = model["negative"].get(word, 0)

        positive_score += pos_value
        negative_score += neg_value

        if pos_value > 0 or neg_value > 0:
            known_words += 1

    # No known words
    if known_words == 0:

        return "neutral", 0.0

    total = positive_score + negative_score

    # Avoid division by zero
    if total == 0:

        return "neutral", 0.0

    # Positive
    if positive_score > negative_score:

        confidence = positive_score / total

        # Low confidence becomes neutral
        if confidence < 0.60:

            return "neutral", confidence

        return "positive", confidence

    # Negative
    elif negative_score > positive_score:

        confidence = negative_score / total

        if confidence < 0.60:

            return "neutral", confidence

        return "negative", confidence

    # Equal evidence
    else:

        return "neutral", 0.5


# --------------------------------------------
# TRAIN
# --------------------------------------------

print()
print("Training model...")

model = train_model(train_set)

print("Training complete!")


# --------------------------------------------
# SAVE MODEL
# --------------------------------------------

with open("sentiment_model_v1.json", "w") as file:

    json.dump(model, file, indent=4)

print("Model saved successfully.")


# --------------------------------------------
# ACCURACY TEST
# --------------------------------------------

print()
print("===================================")
print("          ACCURACY TEST")
print("===================================")

correct = 0

wrong_predictions = []


for text, true_label in test_set:

    guess, confidence = classify(text, model)

    if guess == true_label:

        correct += 1

    else:

        wrong_predictions.append(
            (text, true_label, guess, confidence)
        )


# Calculate accuracy
if len(test_set) > 0:

    accuracy = correct / len(test_set)

else:

    accuracy = 0


print("Correct predictions:", correct)
print("Total test examples:", len(test_set))
print("Accuracy:", round(accuracy * 100, 2), "%")


# --------------------------------------------
# SHOW WRONG PREDICTIONS
# --------------------------------------------

print()
print("===================================")
print("       WRONG PREDICTIONS")
print("===================================")

if len(wrong_predictions) == 0:

    print("No incorrect predictions!")

else:

    for text, true_label, guess, confidence in wrong_predictions:

        print()
        print("Text:", text)
        print("Correct:", true_label)
        print("Agent guessed:", guess)
        print("Confidence:", round(confidence * 100, 2), "%")


# --------------------------------------------
# FINAL INTERACTIVE AGENT
# --------------------------------------------

print()
print("===================================")
print("      SENTIMENT ANALYZER v1.0")
print("===================================")

print("I learned from training examples.")
print("I can report sentiment and confidence.")
print("Type 'quit' to stop.")
print()


while True:

    text = input("You: ")

    if text.lower().strip() == "quit":

        print("Agent: Goodbye!")
        break

    label, confidence = classify(text, model)

    print()

    print("Agent: Sentiment =", label.upper())

    print("Agent: Confidence =", round(confidence * 100), "%")

    # Friendly confidence bar
    bars = int(confidence * 10)

    print(
        "Agent:",
        "[" + "#" * bars + "-" * (10 - bars) + "]"
    )

    print()