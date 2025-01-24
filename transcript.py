class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)
    

def process_transcript(sentence, sentiment_analyzer, sentiment_tracker, transcript_collector):
    sentiment_result = sentiment_analyzer(sentence)[0]
    print(f"Latest Transcript: {sentence}")
    '''print(f"Sentiment: {sentiment_result['label']}, Score: {sentiment_result['score']}")'''

    shift_detected = sentiment_tracker.track_sentiment(sentiment_result['label'])
    transcript_collector.add_part(sentence)

    return sentiment_result, shift_detected