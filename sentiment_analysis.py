from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
class SentimentTracker:
    def __init__(self):
        self.previous_sentiment = None
        self.positive_count = 0
        self.negative_count = 0
        self.positive_to_negative_shifts = 0
        self.negative_to_positive_shifts = 0
        self.shift_detected = None
        self.shift_type = None

    def track_sentiment(self, sentiment):
        shift_detected = "No"  
        self.shift_type = None  

        if self.previous_sentiment:
            if self.previous_sentiment == 'POSITIVE' and sentiment == 'NEGATIVE':
                self.positive_to_negative_shifts += 1
                shift_detected = "Yes"
                self.shift_type = "Positive → Negative" 
            elif self.previous_sentiment == 'NEGATIVE' and sentiment == 'POSITIVE':
                self.negative_to_positive_shifts += 1
                shift_detected = "Yes"
                self.shift_type = "Negative → Positive"

        if sentiment == 'POSITIVE':
            self.positive_count += 1
        elif sentiment == 'NEGATIVE':
            self.negative_count += 1

        self.previous_sentiment = sentiment
        return shift_detected

    def print_summary(self):
        print(f"Positive sentiments: {self.positive_count}")
        print(f"Negative sentiments: {self.negative_count}")
        print(f"Shifts from POSITIVE to NEGATIVE: {self.positive_to_negative_shifts}")
        print(f"Shifts from NEGATIVE to POSITIVE: {self.negative_to_positive_shifts}")
        return {
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "positive_to_negative_shifts": self.positive_to_negative_shifts,
            "negative_to_positive_shifts": self.negative_to_positive_shifts
        }