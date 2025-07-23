from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from prometheus_client import start_http_server, Counter, Gauge
import threading
import time
import os
from multiprocessing import Queue, Process

# Download required NLTK data
nltk.download('vader_lexicon')

def metrics_processor(queue):
    """Separate process to handle metrics updates"""
    from prometheus_client import Counter, Gauge
    metrics = {
        'positive': Counter('sentiment_positive_total', 'Total positive sentiment count'),
        'neutral': Counter('sentiment_neutral_total', 'Total neutral sentiment count'),
        'negative': Counter('sentiment_negative_total', 'Total negative sentiment count'),
        'rows_processed': Counter('sentiment_rows_processed', 'Rows processed count'),
        'processing_time': Gauge('sentiment_processing_seconds', 'Time spent processing')
    }
    
    start_http_server(9091)
    start_time = time.time()
    
    while True:
        message = queue.get()
        if message == 'DONE':
            metrics['processing_time'].set(time.time() - start_time)
            break
        
        sentiment = message
        metrics['rows_processed'].inc(1)
        metrics[sentiment].inc(1)

def analyze_text(text):
    """Sentiment analysis function for UDF"""
    sia = SentimentIntensityAnalyzer()
    if not text or str(text).strip() == '':
        return 'neutral'
    score = sia.polarity_scores(str(text).strip())['compound']
    return 'positive' if score >= 0.05 else 'negative' if score <= -0.05 else 'neutral'

if __name__ == "__main__":
    # Create queue for inter-process communication
    queue = Queue()
    
    # Start metrics processor in separate process
    metric_process = Process(target=metrics_processor, args=(queue,))
    metric_process.start()
    
    try:
        spark = SparkSession.builder \
            .appName("SentimentAnalysis") \
            .getOrCreate()

        # Register UDF
        analyze_udf = udf(analyze_text, StringType())

        # Load and process data
        df = spark.read.csv("data.csv", header=True, escape='"')
        
        # Process each row and send to metrics processor
        for row in df.rdd.toLocalIterator():
            sentiment = analyze_text(row['text'])
            queue.put(sentiment)
        
        # Signal processing complete
        queue.put('DONE')
        
        print("Analysis complete. Metrics available at http://localhost:9091")
        
        # Wait for metrics processor to finish
        metric_process.join()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'spark' in locals():
            spark.stop()
        if metric_process.is_alive():
            metric_process.terminate()
