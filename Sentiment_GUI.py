import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import json
import os
import re
import random
from collections import Counter


# ============================================================
# SENTIMENT ENGINE
# ============================================================

class SentimentEngine:

    def __init__(self):
        self.pos_counts = Counter()
        self.neg_counts = Counter()

        self.training_data = []

        self.total_positive_examples = 0
        self.total_negative_examples = 0

        self.model_trained = False

        self.analysis_history = []

        # Starter knowledge from Week 1.
        # The model can later learn from training data.
        self.positive_words = {
            "good", "great", "love", "happy", "excellent",
            "best", "amazing", "wonderful", "fantastic",
            "awesome", "perfect", "beautiful", "enjoy",
            "enjoyed", "nice", "helpful", "brilliant",
            "success", "successful", "recommend"
        }

        self.negative_words = {
            "bad", "hate", "awful", "terrible", "worst",
            "sad", "horrible", "poor", "disappointing",
            "disappointed", "boring", "useless", "wrong",
            "fail", "failed", "failure", "annoying",
            "angry", "problem", "problems", "avoid"
        }

        self.negation_words = {
            "not", "never", "no", "neither", "nor",
            "isn't", "wasn't", "weren't", "don't",
            "doesn't", "didn't", "can't", "couldn't",
            "won't", "wouldn't"
        }

        # Intensifiers amplify sentiment
        self.intensifiers = {
            "very", "extremely", "incredibly", "so", "really",
            "absolutely", "totally", "completely", "utterly",
            "deeply", "highly", "quite", "rather", "super"
        }

        # Diminishers weaken sentiment
        self.diminishers = {
            "somewhat", "kind", "sort", "rather", "fairly",
            "quite", "slightly", "a bit", "kinda", "sorta"
        }

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    def tokenize(self, text):
        return re.findall(r"[a-zA-Z']+", text.lower())

    # --------------------------------------------------------
    # ADVANCED SENTENCE-BASED SENTIMENT ANALYSIS
    # --------------------------------------------------------

    def analyze(self, text):

        if not text or not text.strip():
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "positive_score": 0,
                "negative_score": 0,
                "positive_words": [],
                "negative_words": [],
                "unknown_words": [],
                "reason": "No text was provided."
            }

        words = self.tokenize(text)

        if not words:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "positive_score": 0,
                "negative_score": 0,
                "positive_words": [],
                "negative_words": [],
                "unknown_words": [],
                "reason": "No valid words found in text."
            }

        positive_found = []
        negative_found = []
        unknown_found = []

        pos = 0
        neg = 0

        # Analyze word by word with context window
        for i, word in enumerate(words):
            # Get context window (previous 2 and next 2 words)
            prev_words = words[max(0, i-2):i]
            next_words = words[i+1:min(len(words), i+3)]

            # Use learned model if available
            learned_pos = self.pos_counts.get(word, 0)
            learned_neg = self.neg_counts.get(word, 0)

            # Otherwise use starter word knowledge
            if learned_pos == 0 and learned_neg == 0:
                if word in self.positive_words:
                    learned_pos = 1
                if word in self.negative_words:
                    learned_neg = 1

            if learned_pos == 0 and learned_neg == 0:
                unknown_found.append(word)
                continue

            # Calculate sentiment weight based on context
            weight = 1.0

            # Check for intensifiers (multiplies sentiment)
            if prev_words and prev_words[-1] in self.intensifiers:
                weight = 2.0
            
            # Check for diminishers (reduces sentiment)
            if prev_words and prev_words[-1] in self.diminishers:
                weight = 0.5

            # Check for negation (flips sentiment)
            is_negated = False
            for prev_word in prev_words:
                if prev_word in self.negation_words:
                    is_negated = True
                    break

            # Flip sentiment if negated
            if is_negated:
                learned_pos, learned_neg = learned_neg, learned_pos

            # Add weighted scores
            if learned_pos > learned_neg:
                pos += learned_pos * weight
                positive_found.append(word)
            elif learned_neg > learned_pos:
                neg += learned_neg * weight
                negative_found.append(word)

        # Remove duplicates while preserving order
        positive_found = list(dict.fromkeys(positive_found))
        negative_found = list(dict.fromkeys(negative_found))

        total = pos + neg

        # Determine sentiment with confidence
        if total == 0:
            sentiment = "neutral"
            confidence = 0.0
            reason = "No sentiment signals were detected. The text appears neutral."

        elif pos > neg:
            sentiment = "positive"
            confidence = min(pos / (pos + neg), 1.0)
            reason = f"Strong positive indicators detected. Positive signals outweigh negative ones."

        elif neg > pos:
            sentiment = "negative"
            confidence = min(neg / (pos + neg), 1.0)
            reason = f"Strong negative indicators detected. Negative signals outweigh positive ones."

        else:
            sentiment = "neutral"
            confidence = 0.5
            reason = "Positive and negative signals are balanced."

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_score": int(pos),
            "negative_score": int(neg),
            "positive_words": positive_found,
            "negative_words": negative_found,
            "unknown_words": unknown_found,
            "reason": reason
        }

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def train(self, data):

        self.training_data = data

        self.pos_counts.clear()
        self.neg_counts.clear()

        self.total_positive_examples = 0
        self.total_negative_examples = 0

        for text, label in data:

            label = label.lower().strip()
            words = self.tokenize(text)

            if label == "positive":
                self.total_positive_examples += 1

                for word in words:
                    self.pos_counts[word] += 1

            elif label == "negative":
                self.total_negative_examples += 1

                for word in words:
                    self.neg_counts[word] += 1

        self.model_trained = True

    # --------------------------------------------------------
    # CLASSIFY USING LEARNED MODEL
    # --------------------------------------------------------

    def classify(self, text):

        result = self.analyze(text)

        return result["sentiment"]

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    def save_model(self, path):

        model = {
            "positive_counts": dict(self.pos_counts),
            "negative_counts": dict(self.neg_counts),
            "training_data": self.training_data,
            "total_positive_examples": self.total_positive_examples,
            "total_negative_examples": self.total_negative_examples
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(model, file, indent=4)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    def load_model(self, path):

        with open(path, "r", encoding="utf-8") as file:
            model = json.load(file)

        self.pos_counts = Counter(model.get("positive_counts", {}))
        self.neg_counts = Counter(model.get("negative_counts", {}))

        self.training_data = [
            tuple(item) for item in model.get("training_data", [])
        ]

        self.total_positive_examples = model.get(
            "total_positive_examples", 0
        )

        self.total_negative_examples = model.get(
            "total_negative_examples", 0
        )

        self.model_trained = True

    # --------------------------------------------------------
    # CSV LOADING
    # --------------------------------------------------------

    def load_csv(self, path):

        data = []

        with open(path, "r", encoding="utf-8-sig", newline="") as file:

            reader = csv.DictReader(file)

            if not reader.fieldnames:
                raise ValueError("CSV does not contain headers.")

            fields = [field.lower().strip() for field in reader.fieldnames]

            text_field = None
            label_field = None

            for original, lower in zip(reader.fieldnames, fields):

                if lower in {"text", "review", "sentence", "content"}:
                    text_field = original

                if lower in {"label", "sentiment", "mood", "target"}:
                    label_field = original

            if text_field is None or label_field is None:
                raise ValueError(
                    "CSV needs a text/review column and a "
                    "label/sentiment column."
                )

            for row in reader:

                text = str(row[text_field]).strip()
                label = str(row[label_field]).strip().lower()

                if label in {"positive", "negative"} and text:
                    data.append((text, label))

        return data


# ============================================================
# APPLICATION
# ============================================================

class SentimentApp:

    BG = "#F8F3EE"
    SIDEBAR = "#E9D7C7"
    CARD = "#FFFDFB"
    CARD_2 = "#F4E9DF"
    PURPLE = "#B88F76"
    PINK = "#D9A89D"
    BLUE = "#A7C3D6"
    WHITE = "#2F241F"
    MUTED = "#6F5A52"
    GREEN = "#A9C7A1"
    RED = "#C78476"
    YELLOW = "#D9BA8B"

    def __init__(self, root):

        self.root = root
        self.engine = SentimentEngine()

        self.current_page = "Dashboard"

        self.stats = {
            "analyses": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }

        self.setup_window()
        self.setup_styles()
        self.build_layout()

        self.show_dashboard()

    # ========================================================
    # WINDOW
    # ========================================================

    def setup_window(self):

        self.root.title("Sentiment Intelligence")
        self.root.geometry("1400x850")
        self.root.minsize(1100, 700)
        self.root.configure(bg=self.BG)

    # ========================================================
    # STYLES
    # ========================================================

    def setup_styles(self):

        style = ttk.Style()

        try:
            style.theme_use("clam")
        except:
            pass

        style.configure(
            "Treeview",
            background=self.CARD,
            foreground=self.WHITE,
            fieldbackground=self.CARD,
            rowheight=34,
            borderwidth=0
        )

        style.configure(
            "Treeview.Heading",
            background=self.CARD_2,
            foreground=self.WHITE,
            font=("Segoe UI", 10, "bold")
        )

        style.map(
            "Treeview",
            background=[("selected", "#D9C3B1")],
            foreground=[("selected", self.WHITE)]
        )

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#EADCCC",
            background=self.PINK,
            bordercolor="#EADCCC",
            lightcolor=self.PINK,
            darkcolor=self.PINK
        )

    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    def build_layout(self):

        self.sidebar = tk.Frame(
            self.root,
            bg=self.SIDEBAR,
            width=245
        )

        self.sidebar.pack(
            side="left",
            fill="y"
        )

        self.sidebar.pack_propagate(False)

        self.main = tk.Frame(
            self.root,
            bg=self.BG
        )

        self.main.pack(
            side="right",
            fill="both",
            expand=True
        )

        self.build_sidebar()

        self.content = tk.Frame(
            self.main,
            bg=self.BG
        )

        self.content.pack(
            fill="both",
            expand=True,
            padx=35,
            pady=25
        )

    # ========================================================
    # SIDEBAR
    # ========================================================

    def build_sidebar(self):

        logo = tk.Frame(
            self.sidebar,
            bg=self.SIDEBAR
        )

        logo.pack(
            fill="x",
            padx=25,
            pady=(30, 35)
        )

        tk.Label(
            logo,
            text="SENTIMENT",
            font=("Segoe UI", 20, "bold"),
            fg=self.WHITE,
            bg=self.SIDEBAR
        ).pack(anchor="w")

        tk.Label(
            logo,
            text="INTELLIGENCE",
            font=("Segoe UI", 8, "bold"),
            fg=self.PINK,
            bg=self.SIDEBAR
        ).pack(anchor="w", pady=(2, 0))

        self.nav_buttons = {}

        navigation = [
            ("⌂", "Dashboard"),
            ("✦", "Analyze"),
            ("◈", "Training"),
            ("▣", "Dataset"),
            ("◉", "Evaluation")
        ]

        for icon, name in navigation:

            button = tk.Button(
                self.sidebar,
                text=f"  {icon}   {name}",
                font=("Segoe UI", 11, "bold"),
                anchor="w",
                relief="flat",
                bd=0,
                cursor="hand2",
                bg=self.SIDEBAR,
                fg=self.MUTED,
                activebackground="#211743",
                activeforeground=self.WHITE,
                padx=25,
                pady=14,
                command=lambda n=name: self.navigate(n)
            )

            button.pack(
                fill="x",
                padx=12,
                pady=2
            )

            self.nav_buttons[name] = button

        # Bottom model status
        bottom = tk.Frame(
            self.sidebar,
            bg=self.CARD,
            highlightthickness=1,
            highlightbackground="#D8C4B1"
        )

        bottom.pack(
            side="bottom",
            fill="x",
            padx=15,
            pady=20
        )

        tk.Label(
            bottom,
            text="MODEL STATUS",
            font=("Segoe UI", 8, "bold"),
            fg=self.MUTED,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=15,
            pady=(13, 3)
        )

        self.model_status = tk.Label(
            bottom,
            text="● Starter knowledge",
            font=("Segoe UI", 10, "bold"),
            fg=self.BLUE,
            bg=self.CARD
        )

        self.model_status.pack(
            anchor="w",
            padx=15,
            pady=(0, 13)
        )

    # ========================================================
    # NAVIGATION
    # ========================================================

    def navigate(self, page):

        self.current_page = page

        for name, button in self.nav_buttons.items():

            if name == page:
                button.configure(
                    bg="#D6B8A5",
                    fg="#FFFDFB"
                )
            else:
                button.configure(
                    bg=self.SIDEBAR,
                    fg=self.MUTED
                )

        self.clear_content()

        if page == "Dashboard":
            self.show_dashboard()

        elif page == "Analyze":
            self.show_analyze()

        elif page == "Training":
            self.show_training()

        elif page == "Dataset":
            self.show_dataset()

        elif page == "Evaluation":
            self.show_evaluation()

    def clear_content(self):

        for widget in self.content.winfo_children():
            widget.destroy()

    # ========================================================
    # COMMON UI
    # ========================================================

    def title(self, heading, subtitle):

        frame = tk.Frame(
            self.content,
            bg=self.BG
        )

        frame.pack(
            fill="x",
            pady=(0, 25)
        )

        tk.Label(
            frame,
            text=heading,
            font=("Segoe UI", 28, "bold"),
            fg=self.WHITE,
            bg=self.BG
        ).pack(anchor="w")

        tk.Label(
            frame,
            text=subtitle,
            font=("Segoe UI", 11),
            fg=self.MUTED,
            bg=self.BG
        ).pack(anchor="w", pady=(4, 0))

    def card(self, parent, width=None, height=None):

        frame = tk.Frame(
            parent,
            bg=self.CARD,
            highlightthickness=1,
            highlightbackground="#E4CDB9"
        )

        if width:
            frame.configure(width=width)

        if height:
            frame.configure(height=height)

        return frame

    def button(
        self,
        parent,
        text,
        command,
        bg=None,
        width=16
    ):

        if bg is None:
            bg = self.PURPLE

        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10, "bold"),
            fg="#FFFDFB",
            bg=bg,
            activebackground="#A27B65",
            activeforeground="#FFFDFB",
            relief="flat",
            bd=0,
            padx=15,
            pady=10,
            cursor="hand2",
            width=width
        )

    # ========================================================
    # DASHBOARD
    # ========================================================

    def show_dashboard(self):

        self.title(
            "Sentiment Intelligence",
            "A modern classifier that learns mood from language."
        )

        stats_frame = tk.Frame(
            self.content,
            bg=self.BG
        )

        stats_frame.pack(
            fill="x"
        )

        self.stat_cards = {}

        cards = [
            ("ANALYSES", "analyses", self.PURPLE),
            ("POSITIVE", "positive", self.GREEN),
            ("NEGATIVE", "negative", self.RED),
            ("NEUTRAL", "neutral", self.BLUE)
        ]

        for label, key, accent in cards:

            c = self.card(stats_frame)

            c.pack(
                side="left",
                fill="both",
                expand=True,
                padx=(0, 12)
            )

            tk.Frame(
                c,
                bg=accent,
                width=5
            ).pack(
                side="left",
                fill="y"
            )

            tk.Label(
                c,
                text=label,
                font=("Segoe UI", 9, "bold"),
                fg=self.MUTED,
                bg=self.CARD
            ).pack(
                anchor="w",
                padx=18,
                pady=(18, 3)
            )

            value = tk.Label(
                c,
                text=str(self.stats[key]),
                font=("Segoe UI", 25, "bold"),
                fg=self.WHITE,
                bg=self.CARD
            )

            value.pack(
                anchor="w",
                padx=18,
                pady=(0, 18)
            )

            self.stat_cards[key] = value

        lower = tk.Frame(
            self.content,
            bg=self.BG
        )

        lower.pack(
            fill="both",
            expand=True,
            pady=(25, 0)
        )

        # System overview
        overview = self.card(lower)

        overview.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 12)
        )

        tk.Label(
            overview,
            text="MODEL OVERVIEW",
            font=("Segoe UI", 11, "bold"),
            fg=self.WHITE,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22,
            pady=(22, 5)
        )

        model_name = (
            "Learned Sentiment Model"
            if self.engine.model_trained
            else "Starter Sentiment Analyzer"
        )

        tk.Label(
            overview,
            text=model_name,
            font=("Segoe UI", 16, "bold"),
            fg=self.PINK,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22
        )

        info = (
            f"Vocabulary: "
            f"{len(set(self.engine.pos_counts) | set(self.engine.neg_counts))}\n\n"
            f"Positive examples: "
            f"{self.engine.total_positive_examples}\n\n"
            f"Negative examples: "
            f"{self.engine.total_negative_examples}\n\n"
            f"Training examples: "
            f"{len(self.engine.training_data)}"
        )

        tk.Label(
            overview,
            text=info,
            font=("Segoe UI", 10),
            fg=self.MUTED,
            bg=self.CARD,
            justify="left"
        ).pack(
            anchor="w",
            padx=22,
            pady=25
        )

        # Quick action card
        actions = self.card(lower)

        actions.pack(
            side="right",
            fill="both",
            expand=True
        )

        tk.Label(
            actions,
            text="QUICK ANALYSIS",
            font=("Segoe UI", 11, "bold"),
            fg=self.WHITE,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22,
            pady=(22, 8)
        )

        self.quick_text = tk.Entry(
            actions,
            font=("Segoe UI", 12),
            bg=self.CARD_2,
            fg=self.WHITE,
            insertbackground=self.WHITE,
            relief="flat"
        )

        self.quick_text.pack(
            fill="x",
            padx=22,
            ipady=12
        )

        self.button(
            actions,
            "Analyze Text →",
            self.quick_analyze,
            bg=self.PURPLE
        ).pack(
            anchor="w",
            padx=22,
            pady=18
        )

        tk.Label(
            actions,
            text="Try:  I absolutely love this amazing product!",
            font=("Segoe UI", 9),
            fg=self.MUTED,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22
        )

    def update_dashboard_stats(self):

        if hasattr(self, "stat_cards"):

            for key, label in self.stat_cards.items():

                label.configure(
                    text=str(self.stats[key])
                )

    # ========================================================
    # ANALYZE PAGE
    # ========================================================

    def show_analyze(self):

        self.title(
            "Analyze Text",
            "Read a sentence, classify its mood, and inspect the evidence."
        )

        container = tk.Frame(
            self.content,
            bg=self.BG
        )

        container.pack(
            fill="both",
            expand=True
        )

        container.grid_columnconfigure(0, weight=2)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)

        # LEFT
        left = self.card(container)

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 12)
        )

        tk.Label(
            left,
            text="YOUR TEXT",
            font=("Segoe UI", 10, "bold"),
            fg=self.MUTED,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22,
            pady=(22, 8)
        )

        self.analyze_text = tk.Text(
            left,
            height=12,
            font=("Segoe UI", 13),
            bg=self.CARD_2,
            fg=self.WHITE,
            insertbackground=self.WHITE,
            relief="flat",
            wrap="word",
            padx=15,
            pady=15
        )

        self.analyze_text.pack(
            fill="both",
            expand=True,
            padx=22
        )

        self.analyze_text.bind(
            "<KeyRelease>",
            lambda event: self.perform_analysis()
        )

        self.analyze_text.bind(
            "<Control-Return>",
            lambda event: self.perform_analysis()
        )

        controls = tk.Frame(
            left,
            bg=self.CARD
        )

        controls.pack(
            fill="x",
            padx=22,
            pady=20
        )

        self.button(
            controls,
            "✦ Analyze",
            self.perform_analysis
        ).pack(
            side="left"
        )

        self.button(
            controls,
            "Clear",
            self.clear_analysis,
            bg="#D8BBA0"
        ).pack(
            side="left",
            padx=10
        )

        # RIGHT
        right = self.card(container)

        right.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        tk.Label(
            right,
            text="ANALYSIS RESULT",
            font=("Segoe UI", 10, "bold"),
            fg=self.MUTED,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22,
            pady=(22, 12)
        )

        self.analysis_question = tk.Label(
            right,
            text="No question yet.",
            font=("Segoe UI", 12, "bold"),
            fg=self.WHITE,
            bg=self.CARD,
            justify="left",
            wraplength=430
        )

        self.analysis_question.pack(
            anchor="w",
            padx=22,
            pady=(0, 10)
        )

        self.result_sentiment = tk.Label(
            right,
            text="READY",
            font=("Segoe UI", 30, "bold"),
            fg=self.PINK,
            bg=self.CARD
        )

        self.result_sentiment.pack(
            anchor="w",
            padx=22
        )

        self.confidence_label = tk.Label(
            right,
            text="Confidence — 0%",
            font=("Segoe UI", 11, "bold"),
            fg=self.WHITE,
            bg=self.CARD
        )

        self.confidence_label.pack(
            anchor="w",
            padx=22,
            pady=(10, 6)
        )

        self.confidence_bar = ttk.Progressbar(
            right,
            style="Horizontal.TProgressbar",
            maximum=100
        )

        self.confidence_bar.pack(
            fill="x",
            padx=22
        )

        self.scores_label = tk.Label(
            right,
            text="",
            font=("Segoe UI", 10),
            fg=self.MUTED,
            bg=self.CARD,
            justify="left"
        )

        self.scores_label.pack(
            anchor="w",
            padx=22,
            pady=18
        )

        self.signal_label = tk.Label(
            right,
            text="",
            font=("Segoe UI", 10),
            fg=self.WHITE,
            bg=self.CARD,
            justify="left",
            wraplength=450
        )

        self.signal_label.pack(
            anchor="w",
            padx=22,
            pady=10
        )

        self.reason_label = tk.Label(
            right,
            text="",
            font=("Segoe UI", 10),
            fg=self.MUTED,
            bg=self.CARD,
            justify="left",
            wraplength=450
        )

        self.reason_label.pack(
            anchor="w",
            padx=22,
            pady=8
        )

    def perform_analysis(self):

        text = self.analyze_text.get(
            "1.0",
            "end"
        ).strip()

        result = self.engine.analyze(text)

        sentiment = result["sentiment"]
        confidence = result["confidence"]

        self.engine.analysis_history.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence
        })

        if sentiment == "positive":
            display = "POSITIVE"
            accent = self.GREEN

        elif sentiment == "negative":
            display = "NEGATIVE"
            accent = self.RED

        else:
            display = "NEUTRAL"
            accent = self.BLUE

        self.result_sentiment.configure(
            text=display,
            fg=accent
        )

        self.confidence_label.configure(
            text=f"Confidence — {round(confidence * 100)}%"
        )

        self.confidence_bar["value"] = confidence * 100

        self.analysis_question.configure(
            text=text
        )

        positive_words = result.get("positive_words", [])
        negative_words = result.get("negative_words", [])
        pos_score = result.get("positive_score", 0)
        neg_score = result.get("negative_score", 0)

        scores_text = f"Positive: {pos_score} | Negative: {neg_score}"
        self.scores_label.configure(
            text=scores_text
        )

        signal_parts = []
        if positive_words:
            signal_parts.append(f"✓ Positive: {', '.join(positive_words)}")
        if negative_words:
            signal_parts.append(f"✗ Negative: {', '.join(negative_words)}")

        signal_text = "\n".join(signal_parts) if signal_parts else "No sentiment signals detected."
        self.signal_label.configure(
            text=signal_text
        )

        reason = result.get("reason", "")
        self.reason_label.configure(
            text=reason
        )

    def clear_analysis(self):

        self.analyze_text.delete(
            "1.0",
            "end"
        )

        self.result_sentiment.configure(
            text="READY",
            fg=self.PINK
        )

        self.confidence_label.configure(
            text="Confidence — 0%"
        )

        self.confidence_bar["value"] = 0

        self.analysis_question.configure(
            text="No question yet."
        )

        self.scores_label.configure(
            text=""
        )

        self.signal_label.configure(
            text=""
        )

        self.reason_label.configure(
            text=""
        )

    def quick_analyze(self):

        text = self.quick_text.get().strip()

        if not text:
            return

        self.navigate("Analyze")

        self.analyze_text.insert(
            "1.0",
            text
        )

        self.perform_analysis()

    # ========================================================
    # TRAINING PAGE
    # ========================================================

    def show_training(self):

        self.title(
            "Model Training",
            "Teach the agent word moods from labelled examples."
        )

        top = self.card(self.content)

        top.pack(
            fill="x",
            pady=(0, 15)
        )

        tk.Label(
            top,
            text="TRAINING DATA",
            font=("Segoe UI", 10, "bold"),
            fg=self.MUTED,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22,
            pady=(20, 5)
        )

        tk.Label(
            top,
            text=(
                "Load a CSV containing text/review and "
                "positive/negative sentiment labels."
            ),
            font=("Segoe UI", 11),
            fg=self.WHITE,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22
        )

        buttons = tk.Frame(
            top,
            bg=self.CARD
        )

        buttons.pack(
            anchor="w",
            padx=22,
            pady=18
        )

        self.button(
            buttons,
            "＋ Load CSV",
            self.load_training_csv
        ).pack(
            side="left"
        )

        self.button(
            buttons,
            "⚡ Train Model",
            self.train_model,
            bg=self.PINK
        ).pack(
            side="left",
            padx=10
        )

        self.button(
            buttons,
            "Save Model",
            self.save_model,
            bg="#D8BBA0"
        ).pack(
            side="left"
        )

        self.button(
            buttons,
            "Load Model",
            self.load_model,
            bg="#D8BBA0"
        ).pack(
            side="left",
            padx=10
        )

        # Statistics
        stats = tk.Frame(
            self.content,
            bg=self.BG
        )

        stats.pack(
            fill="x",
            pady=(0, 15)
        )

        self.training_info = []

        for label, value in [
            ("Examples", "0"),
            ("Positive", "0"),
            ("Negative", "0"),
            ("Vocabulary", "0")
        ]:

            c = self.card(stats)

            c.pack(
                side="left",
                fill="both",
                expand=True,
                padx=(0, 10)
            )

            tk.Label(
                c,
                text=label.upper(),
                font=("Segoe UI", 8, "bold"),
                fg=self.MUTED,
                bg=self.CARD
            ).pack(
                anchor="w",
                padx=15,
                pady=(15, 2)
            )

            value_label = tk.Label(
                c,
                text=value,
                font=("Segoe UI", 20, "bold"),
                fg=self.WHITE,
                bg=self.CARD
            )

            value_label.pack(
                anchor="w",
                padx=15,
                pady=(0, 15)
            )

            self.training_info.append(value_label)

        # Vocabulary
        vocab_card = self.card(self.content)

        vocab_card.pack(
            fill="both",
            expand=True
        )

        tk.Label(
            vocab_card,
            text="LEARNED WORD SIGNALS",
            font=("Segoe UI", 10, "bold"),
            fg=self.WHITE,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=20,
            pady=18
        )

        self.vocab_text = tk.Text(
            vocab_card,
            bg=self.CARD_2,
            fg=self.MUTED,
            font=("Consolas", 10),
            relief="flat",
            wrap="word"
        )

        self.vocab_text.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=(0, 20)
        )

        self.refresh_training_page()

    def load_training_csv(self):

        path = filedialog.askopenfilename(
            title="Select Training CSV",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:

            data = self.engine.load_csv(path)

            self.engine.training_data = data

            self.refresh_training_page()

            messagebox.showinfo(
                "Dataset Loaded",
                f"Loaded {len(data)} labelled examples."
            )

        except Exception as error:

            messagebox.showerror(
                "CSV Error",
                str(error)
            )

    def train_model(self):

        if not self.engine.training_data:

            messagebox.showwarning(
                "No Training Data",
                "Load a labelled CSV dataset first."
            )

            return

        self.engine.train(
            self.engine.training_data
        )

        self.model_status.configure(
            text="● Learned model active",
            fg=self.GREEN
        )

        self.refresh_training_page()

        messagebox.showinfo(
            "Training Complete",
            (
                "The sentiment model has learned word "
                "signals from your training data."
            )
        )

    def refresh_training_page(self):

        if not hasattr(self, "training_info"):
            return

        data = self.engine.training_data

        positive = sum(
            1 for _, label in data
            if label == "positive"
        )

        negative = sum(
            1 for _, label in data
            if label == "negative"
        )

        vocabulary = len(
            set(self.engine.pos_counts)
            | set(self.engine.neg_counts)
        )

        values = [
            len(data),
            positive,
            negative,
            vocabulary
        ]

        for label, value in zip(
            self.training_info,
            values
        ):
            label.configure(
                text=str(value)
            )

        if hasattr(self, "vocab_text"):

            self.vocab_text.delete(
                "1.0",
                "end"
            )

            if not vocabulary:

                self.vocab_text.insert(
                    "end",
                    "No learned vocabulary yet."
                )

                return

            all_words = (
                set(self.engine.pos_counts)
                | set(self.engine.neg_counts)
            )

            rows = []

            for word in all_words:

                p = self.engine.pos_counts.get(word, 0)
                n = self.engine.neg_counts.get(word, 0)

                if p > n:
                    mood = "POSITIVE"

                elif n > p:
                    mood = "NEGATIVE"

                else:
                    mood = "BALANCED"

                rows.append(
                    (word, p, n, mood)
                )

            rows.sort(
                key=lambda item: item[0]
            )

            self.vocab_text.insert(
                "end",
                f"{'WORD':<20}"
                f"{'POS':<8}"
                f"{'NEG':<8}"
                f"MOOD\n"
            )

            self.vocab_text.insert(
                "end",
                "-" * 55 + "\n"
            )

            for word, p, n, mood in rows[:500]:

                self.vocab_text.insert(
                    "end",
                    f"{word:<20}"
                    f"{p:<8}"
                    f"{n:<8}"
                    f"{mood}\n"
                )

    def save_model(self):

        if not self.engine.model_trained:

            messagebox.showwarning(
                "Model Not Trained",
                "Train the model before saving it."
            )

            return

        path = filedialog.asksaveasfilename(
            title="Save Sentiment Model",
            defaultextension=".json",
            filetypes=[
                ("JSON model", "*.json")
            ]
        )

        if not path:
            return

        try:

            self.engine.save_model(path)

            messagebox.showinfo(
                "Model Saved",
                "Your learned model was saved successfully."
            )

        except Exception as error:

            messagebox.showerror(
                "Save Error",
                str(error)
            )

    def load_model(self):

        path = filedialog.askopenfilename(
            title="Load Sentiment Model",
            filetypes=[
                ("JSON model", "*.json"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:

            self.engine.load_model(path)

            self.model_status.configure(
                text="● Learned model active",
                fg=self.GREEN
            )

            self.refresh_training_page()

            messagebox.showinfo(
                "Model Loaded",
                "Learned sentiment model loaded successfully."
            )

        except Exception as error:

            messagebox.showerror(
                "Load Error",
                str(error)
            )

    # ========================================================
    # DATASET PAGE
    # ========================================================

    def show_dataset(self):

        self.title(
            "Dataset Explorer",
            "Inspect the labelled examples used to teach the classifier."
        )

        toolbar = tk.Frame(
            self.content,
            bg=self.BG
        )

        toolbar.pack(
            fill="x",
            pady=(0, 15)
        )

        self.button(
            toolbar,
            "＋ Import CSV",
            self.load_training_csv
        ).pack(
            side="left"
        )

        count = len(
            self.engine.training_data
        )

        tk.Label(
            toolbar,
            text=f"  {count} labelled examples loaded",
            font=("Segoe UI", 10),
            fg=self.MUTED,
            bg=self.BG
        ).pack(
            side="left",
            padx=10
        )

        table_card = self.card(
            self.content
        )

        table_card.pack(
            fill="both",
            expand=True
        )

        columns = (
            "number",
            "label",
            "text"
        )

        tree = ttk.Treeview(
            table_card,
            columns=columns,
            show="headings"
        )

        tree.heading(
            "number",
            text="#"
        )

        tree.heading(
            "label",
            text="SENTIMENT"
        )

        tree.heading(
            "text",
            text="TEXT / REVIEW"
        )

        tree.column(
            "number",
            width=60
        )

        tree.column(
            "label",
            width=130
        )

        tree.column(
            "text",
            width=700
        )

        scrollbar = ttk.Scrollbar(
            table_card,
            orient="vertical",
            command=tree.yview
        )

        tree.configure(
            yscrollcommand=scrollbar.set
        )

        tree.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(18, 0),
            pady=18
        )

        scrollbar.pack(
            side="right",
            fill="y",
            padx=(0, 18),
            pady=18
        )

        for index, (text, label) in enumerate(
            self.engine.training_data,
            start=1
        ):

            tree.insert(
                "",
                "end",
                values=(
                    index,
                    label.upper(),
                    text
                )
            )

    # ========================================================
    # EVALUATION
    # ========================================================

    def show_evaluation(self):

        self.title(
            "Model Evaluation",
            "Test the classifier on data it did not train on."
        )

        controls = self.card(
            self.content
        )

        controls.pack(
            fill="x",
            pady=(0, 15)
        )

        tk.Label(
            controls,
            text="TEST CONFIGURATION",
            font=("Segoe UI", 10, "bold"),
            fg=self.MUTED,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22,
            pady=(20, 5)
        )

        tk.Label(
            controls,
            text=(
                "The evaluator uses an 80/20 train-test split, "
                "as described in the lab guide."
            ),
            font=("Segoe UI", 11),
            fg=self.WHITE,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22
        )

        self.button(
            controls,
            "▶ Run Evaluation",
            self.run_evaluation,
            bg=self.PINK
        ).pack(
            anchor="w",
            padx=22,
            pady=18
        )

        results = tk.Frame(
            self.content,
            bg=self.BG
        )

        results.pack(
            fill="both",
            expand=True
        )

        # Accuracy card
        accuracy_card = self.card(
            results
        )

        accuracy_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 12)
        )

        tk.Label(
            accuracy_card,
            text="ACCURACY",
            font=("Segoe UI", 10, "bold"),
            fg=self.MUTED,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22,
            pady=(25, 5)
        )

        self.accuracy_value = tk.Label(
            accuracy_card,
            text="—",
            font=("Segoe UI", 48, "bold"),
            fg=self.GREEN,
            bg=self.CARD
        )

        self.accuracy_value.pack(
            anchor="w",
            padx=22
        )

        self.evaluation_summary = tk.Label(
            accuracy_card,
            text="Run an evaluation to measure performance.",
            font=("Segoe UI", 10),
            fg=self.MUTED,
            bg=self.CARD,
            justify="left"
        )

        self.evaluation_summary.pack(
            anchor="w",
            padx=22,
            pady=20
        )

        # Error analysis
        error_card = self.card(
            results
        )

        error_card.pack(
            side="right",
            fill="both",
            expand=True
        )

        tk.Label(
            error_card,
            text="ERROR ANALYSIS",
            font=("Segoe UI", 10, "bold"),
            fg=self.MUTED,
            bg=self.CARD
        ).pack(
            anchor="w",
            padx=22,
            pady=(25, 8)
        )

        self.errors_text = tk.Text(
            error_card,
            bg=self.CARD_2,
            fg=self.WHITE,
            font=("Segoe UI", 10),
            relief="flat",
            wrap="word"
        )

        self.errors_text.pack(
            fill="both",
            expand=True,
            padx=22,
            pady=(0, 22)
        )

    def run_evaluation(self):

        data = self.engine.training_data

        if len(data) < 2:

            messagebox.showwarning(
                "Not Enough Data",
                "Load at least two labelled examples."
            )

            return

        shuffled = data.copy()

        random.Random(42).shuffle(
            shuffled
        )

        split = max(
            1,
            int(len(shuffled) * 0.8)
        )

        train_set = shuffled[:split]
        test_set = shuffled[split:]

        if not test_set:

            messagebox.showwarning(
                "Test Set Empty",
                "Add more labelled examples."
            )

            return

        # Train ONLY on training set
        temporary_pos = self.engine.pos_counts.copy()
        temporary_neg = self.engine.neg_counts.copy()
        temporary_data = self.engine.training_data.copy()
        temporary_model = self.engine.model_trained

        self.engine.train(
            train_set
        )

        correct = 0
        errors = []

        for text, true_label in test_set:

            prediction = self.engine.classify(
                text
            )

            if prediction == true_label:
                correct += 1

            else:
                errors.append(
                    (
                        text,
                        true_label,
                        prediction
                    )
                )

        accuracy = correct / len(test_set)

        # Restore full model state
        self.engine.pos_counts = temporary_pos
        self.engine.neg_counts = temporary_neg
        self.engine.training_data = temporary_data
        self.engine.model_trained = temporary_model

        self.accuracy_value.configure(
            text=f"{round(accuracy * 100)}%"
        )

        self.evaluation_summary.configure(
            text=(
                f"Training examples: {len(train_set)}\n"
                f"Testing examples: {len(test_set)}\n"
                f"Correct predictions: {correct}\n"
                f"Incorrect predictions: {len(errors)}"
            )
        )

        self.errors_text.delete(
            "1.0",
            "end"
        )

        if not errors:

            self.errors_text.insert(
                "end",
                "✓ No incorrect predictions "
                "were found in the test set."
            )

        else:

            for number, (
                text,
                true_label,
                prediction
            ) in enumerate(
                errors,
                start=1
            ):

                self.errors_text.insert(
                    "end",
                    (
                        f"ERROR {number}\n"
                        f"Text: {text}\n"
                        f"Expected: {true_label.upper()}\n"
                        f"Predicted: {prediction.upper()}\n"
                        f"{'-' * 50}\n\n"
                    )
                )

    # ========================================================
    # APPLICATION EXIT
    # ========================================================

    def on_close(self):

        answer = messagebox.askyesno(
            "Exit NOVA",
            "Are you sure you want to close the application?"
        )

        if answer:
            self.root.destroy()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = SentimentApp(root)

    root.protocol(
        "WM_DELETE_WINDOW",
        app.on_close
    )

    root.mainloop()