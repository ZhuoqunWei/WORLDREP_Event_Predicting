import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from datetime import datetime
from tqdm import tqdm

def load_worldrep_data(file_path):
    """
    Load the WORLDREP dataset from CSV file
    """
    print(f"Loading data from {file_path}...")
    try:
        # Try with utf-8 encoding first
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try with latin1 encoding if utf-8 fails
        df = pd.read_csv(file_path, encoding='latin1')
    
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Display some basic info
    print("\nColumn names:", df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

def clean_text(text):
    """Clean and normalize text content"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def basic_sentiment_analysis(text):
    """
    Very simple lexicon-based sentiment analysis
    Returns a score between -1 and 1
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # Simple positive and negative word lists
    positive_words = [
        'good', 'great', 'excellent', 'positive', 'nice', 'wonderful', 'love', 
        'happy', 'joy', 'success', 'successful', 'benefit', 'benefits', 'agreement',
        'peace', 'peaceful', 'cooperation', 'cooperate', 'ally', 'allies', 'friend',
        'friendly', 'support', 'supporting', 'supported', 'achieve', 'achievement'
    ]
    
    negative_words = [
        'bad', 'terrible', 'horrible', 'negative', 'awful', 'hate', 'sad', 
        'unhappy', 'failure', 'fail', 'problem', 'threat', 'threatening', 'danger',
        'dangerous', 'conflict', 'war', 'attack', 'enemy', 'enemies', 'hostile',
        'hostility', 'tension', 'crisis', 'oppose', 'opposition', 'sanction', 'sanctions'
    ]
    
    # Count word occurrences
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Calculate total words
    total_words = len(words)
    if total_words == 0:
        return 0.0
    
    # Calculate sentiment score
    sentiment_score = (positive_count - negative_count) / (positive_count + negative_count + 0.001)
    
    return sentiment_score

def preprocess_worldrep_data(input_file, output_file="processed_worldrep.csv"):
    """
    Preprocess the WORLDREP dataset without using NLTK
    """
    # Load data
    df = load_worldrep_data(input_file)
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Step 1: Handle missing values
    print("\nHandling missing values...")
    
    # Fill missing Score values (if any)
    if 'Score' in processed_df.columns:
        # Check for missing values
        missing_scores = processed_df['Score'].isnull().sum()
        if missing_scores > 0:
            print(f"Found {missing_scores} missing Score values")
            # We'll keep NaN values but note their presence
    
    # Step 2: Convert DATE to datetime
    print("\nConverting dates...")
    if 'DATE' in processed_df.columns:
        processed_df['DATE'] = pd.to_datetime(processed_df['DATE'], format='%Y%m%d%H%M%S')
        
        # Extract temporal features
        processed_df['Year'] = processed_df['DATE'].dt.year
        processed_df['Month'] = processed_df['DATE'].dt.month
        processed_df['Day'] = processed_df['DATE'].dt.day
        processed_df['DayOfWeek'] = processed_df['DATE'].dt.dayofweek
    
    # Step 3: Clean and analyze text content
    print("\nProcessing text content...")
    if 'CONTENT' in processed_df.columns:
        # Clean text
        processed_df['CleanedContent'] = processed_df['CONTENT'].apply(clean_text)
        
        # Extract text length as a feature
        processed_df['ContentLength'] = processed_df['CONTENT'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        
        # Count mentions of country names in content (simple approach)
        processed_df['Country1Mentions'] = processed_df.apply(
            lambda row: row['CONTENT'].lower().count(row['Country1'].lower()) if isinstance(row['CONTENT'], str) and isinstance(row['Country1'], str) else 0, 
            axis=1
        )
        
        processed_df['Country2Mentions'] = processed_df.apply(
            lambda row: row['CONTENT'].lower().count(row['Country2'].lower()) if isinstance(row['CONTENT'], str) and isinstance(row['Country2'], str) else 0, 
            axis=1
        )
    
    # Step 4: Apply simple sentiment analysis
    print("\nApplying basic sentiment analysis...")
    # Apply simple sentiment analysis
    tqdm.pandas(desc="Analyzing sentiment")
    processed_df['SimpleSentiment'] = processed_df['CleanedContent'].progress_apply(basic_sentiment_analysis)
    
    # Step 5: Create country-pair features
    print("\nCreating country relationship features...")
    # Create a combined feature of country pairs (alphabetically sorted)
    processed_df['CountryPair'] = processed_df.apply(
        lambda row: '+'.join(sorted([str(row['Country1']), str(row['Country2'])])), 
        axis=1
    )
    
    # Count article frequency by country pair
    country_pair_counts = processed_df['CountryPair'].value_counts()
    processed_df['CountryPairFrequency'] = processed_df['CountryPair'].map(country_pair_counts)
    
    # Step 6: Basic data exploration and visualization
    print("\nGenerating basic statistics and visualizations...")
    
    # Analyze sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(processed_df['SimpleSentiment'], kde=True)
    plt.title('Distribution of Simple Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.savefig('sentiment_distribution.png')
    print("Sentiment distribution plot saved as 'sentiment_distribution.png'")
    
    # Analyze Score distribution (if available)
    if 'Score' in processed_df.columns and processed_df['Score'].notna().any():
        plt.figure(figsize=(10, 6))
        sns.histplot(processed_df['Score'].dropna(), kde=True)
        plt.title('Distribution of Country Relationship Scores')
        plt.xlabel('Score (0=Cooperation, 1=Conflict)')
        plt.ylabel('Frequency')
        plt.savefig('score_distribution.png')
        print("Score distribution plot saved as 'score_distribution.png'")
    
    # Plot the relationship between sentiment and conflict score
    if 'Score' in processed_df.columns and processed_df['Score'].notna().any():
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='SimpleSentiment', y='Score', data=processed_df.dropna(subset=['Score']))
        plt.title('Relationship Between Article Sentiment and Conflict Score')
        plt.xlabel('Simple Sentiment Score (-1 to 1)')
        plt.ylabel('Conflict Score (0 to 1)')
        plt.savefig('sentiment_vs_conflict.png')
        print("Sentiment vs. conflict plot saved as 'sentiment_vs_conflict.png'")
    
    # Step 7: Feature engineering for time series analysis
    print("\nCreating time series features...")
    
    # Group by date and country pair to get daily sentiment and event counts
    time_features = processed_df.groupby(['DATE', 'CountryPair']).agg({
        'SimpleSentiment': 'mean',
        'EventID': 'count',
        'Score': 'mean'
    }).reset_index()
    
    time_features.rename(columns={'EventID': 'DailyEventCount'}, inplace=True)
    
    # Save time series features separately
    time_features.to_csv('time_series_features.csv', index=False)
    print("Time series features saved to 'time_series_features.csv'")
    
    # Step 8: Save preprocessed data
    print(f"\nSaving preprocessed data to {output_file}...")
    processed_df.to_csv(output_file, index=False)
    
    # Save a sample for inspection
    sample_file = "sample_" + output_file
    processed_df.sample(min(1000, len(processed_df))).to_csv(sample_file, index=False)
    print(f"Sample saved to {sample_file}")
    
    print("\nPreprocessing completed successfully!")
    return processed_df

if __name__ == "__main__":
    # Set the input file path
    input_file = "data/worldrep_dataset_v2.csv"
    
    # Process the data
    processed_data = preprocess_worldrep_data(input_file)
    
    # Display the final dataset shape and columns
    print("\nFinal dataset shape:", processed_data.shape)
    print("\nColumns in preprocessed data:")
    for col in processed_data.columns:
        print(f"- {col}")