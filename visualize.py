import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import calendar
import os

# Create output directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

def load_processed_data(file_path="processed_worldrep.csv"):
    """
    Load the preprocessed WORLDREP dataset
    """
    print(f"Loading processed data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert DATE back to datetime if needed
    if 'DATE' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['DATE']):
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    return df

def visualize_sentiment_distribution(df, sentiment_col='SimpleSentiment', output_dir='visualizations'):
    """
    Visualize the distribution of sentiment scores
    """
    plt.figure(figsize=(12, 8))
    
    # Create a custom color palette
    palette = sns.color_palette("coolwarm", as_cmap=True)
    
    # Plot the distribution with KDE
    ax = sns.histplot(df[sentiment_col], kde=True, bins=50, color='skyblue')
    
    # Add vertical line at 0
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Add annotations for negative, neutral, and positive regions
    y_max = ax.get_ylim()[1]
    plt.text(-0.75, y_max * 0.8, 'Negative', fontsize=14, ha='center')
    plt.text(0, y_max * 0.8, 'Neutral', fontsize=14, ha='center')
    plt.text(0.75, y_max * 0.8, 'Positive', fontsize=14, ha='center')
    
    # Add descriptive statistics
    mean_val = df[sentiment_col].mean()
    median_val = df[sentiment_col].median()
    std_val = df[sentiment_col].std()
    
    stats_text = f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd Dev: {std_val:.3f}"
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                 fontsize=12, ha='right', va='top',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Set labels and title
    plt.xlabel('Sentiment Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Sentiment Scores in WORLDREP Dataset', fontsize=16)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'sentiment_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sentiment distribution plot to {output_path}")

def visualize_conflict_score_distribution(df, output_dir='visualizations'):
    """
    Visualize the distribution of conflict scores
    """
    if 'Score' not in df.columns or df['Score'].isna().all():
        print("Warning: Conflict score column not found or all values are NaN")
        return
    
    # Filter out NaN values
    conflict_df = df.dropna(subset=['Score'])
    
    plt.figure(figsize=(12, 8))
    
    # Create custom colormap from green to red
    colors = [(0.0, 0.8, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]  # Green to Yellow to Red
    cmap = LinearSegmentedColormap.from_list('cooperation_conflict', colors, N=100)
    
    # Plot histogram with KDE
    ax = sns.histplot(conflict_df['Score'], kde=True, bins=50, color='skyblue')
    
    # Add annotations for cooperation vs conflict
    y_max = ax.get_ylim()[1]
    plt.text(0.1, y_max * 0.8, 'Cooperation', fontsize=14, ha='center', color='green')
    plt.text(0.9, y_max * 0.8, 'Conflict', fontsize=14, ha='center', color='red')
    
    # Add descriptive statistics
    mean_val = conflict_df['Score'].mean()
    median_val = conflict_df['Score'].median()
    std_val = conflict_df['Score'].std()
    
    stats_text = f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd Dev: {std_val:.3f}"
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                 fontsize=12, ha='right', va='top',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Set labels and title
    plt.xlabel('Conflict Score (0 = Cooperation, 1 = Conflict)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Country Relationship Scores', fontsize=16)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'conflict_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved conflict score distribution plot to {output_path}")

def visualize_sentiment_vs_conflict(df, sentiment_col='SimpleSentiment', output_dir='visualizations'):
    """
    Visualize the relationship between sentiment and conflict score
    """
    if 'Score' not in df.columns or df['Score'].isna().all():
        print("Warning: Conflict score column not found or all values are NaN")
        return
    
    # Filter out NaN values
    filtered_df = df.dropna(subset=[sentiment_col, 'Score'])
    
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot with hexbin for density
    plt.hexbin(filtered_df[sentiment_col], filtered_df['Score'], 
              gridsize=30, cmap='viridis', alpha=0.8)
    
    # Add a colorbar
    cb = plt.colorbar()
    cb.set_label('Count', fontsize=12)
    
    # Add a trend line
    try:
        # Calculate correlation
        correlation = filtered_df[sentiment_col].corr(filtered_df['Score'])
        
        # Add regression line
        sns.regplot(x=sentiment_col, y='Score', data=filtered_df, 
                   scatter=False, color='red', line_kws={"linestyle":"--"})
        
        # Add correlation annotation
        plt.annotate(f"Correlation: {correlation:.3f}", 
                    xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=12, ha='left', va='top',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    except:
        print("Warning: Could not calculate correlation or trendline")
    
    # Set labels and title
    plt.xlabel('Sentiment Score', fontsize=14)
    plt.ylabel('Conflict Score (0 = Cooperation, 1 = Conflict)', fontsize=14)
    plt.title('Relationship Between Article Sentiment and Country Conflict Score', fontsize=16)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'sentiment_vs_conflict.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sentiment vs conflict plot to {output_path}")

def visualize_sentiment_time_series(df, sentiment_col='SimpleSentiment', output_dir='visualizations'):
    """
    Visualize sentiment trends over time
    """
    if 'DATE' not in df.columns:
        print("Warning: DATE column not found")
        return
    
    # Ensure DATE is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['DATE']):
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Group by date and calculate average sentiment
    daily_sentiment = df.groupby(df['DATE'].dt.date)[sentiment_col].mean().reset_index()
    
    plt.figure(figsize=(16, 8))
    
    # Plot sentiment over time
    plt.plot(daily_sentiment['DATE'], daily_sentiment[sentiment_col], 
            marker='o', markersize=3, linestyle='-', linewidth=1.5, color='blue', alpha=0.7)
    
    # Add smoothed trend line
    try:
        window_size = min(30, len(daily_sentiment) // 5)  # Adjust window size based on data
        if window_size > 1:
            daily_sentiment['RollingMean'] = daily_sentiment[sentiment_col].rolling(window=window_size, center=True).mean()
            plt.plot(daily_sentiment['DATE'], daily_sentiment['RollingMean'], 
                    linestyle='-', linewidth=3, color='red', alpha=0.8, label=f'{window_size}-day Moving Average')
    except:
        print("Warning: Could not calculate rolling average")
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()
    
    # Add zero reference line
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Average Sentiment Score', fontsize=14)
    plt.title('Sentiment Trend Over Time', fontsize=16)
    
    # Add legend if we created a rolling mean
    if 'RollingMean' in daily_sentiment.columns:
        plt.legend(fontsize=12)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'sentiment_time_series.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sentiment time series plot to {output_path}")

def visualize_conflict_time_series(df, output_dir='visualizations'):
    """
    Visualize conflict score trends over time
    """
    if 'Score' not in df.columns or df['Score'].isna().all():
        print("Warning: Conflict score column not found or all values are NaN")
        return
    
    if 'DATE' not in df.columns:
        print("Warning: DATE column not found")
        return
    
    # Ensure DATE is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['DATE']):
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Group by date and calculate average conflict score
    daily_conflict = df.dropna(subset=['Score']).groupby(df['DATE'].dt.date)['Score'].mean().reset_index()
    
    plt.figure(figsize=(16, 8))
    
    # Plot conflict over time
    plt.plot(daily_conflict['DATE'], daily_conflict['Score'], 
            marker='o', markersize=3, linestyle='-', linewidth=1.5, color='darkred', alpha=0.7)
    
    # Add smoothed trend line
    try:
        window_size = min(30, len(daily_conflict) // 5)  # Adjust window size based on data
        if window_size > 1:
            daily_conflict['RollingMean'] = daily_conflict['Score'].rolling(window=window_size, center=True).mean()
            plt.plot(daily_conflict['DATE'], daily_conflict['RollingMean'], 
                   linestyle='-', linewidth=3, color='blue', alpha=0.8, label=f'{window_size}-day Moving Average')
    except:
        print("Warning: Could not calculate rolling average")
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()
    
    # Add reference line at 0.5 (midpoint between cooperation and conflict)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Average Conflict Score', fontsize=14)
    plt.title('Country Relationship Conflict Trend Over Time', fontsize=16)
    
    # Add colorbar-like legend for cooperation vs conflict
    import matplotlib.patches as mpatches
    cooperation_patch = mpatches.Patch(color='green', label='Cooperation')
    conflict_patch = mpatches.Patch(color='red', label='Conflict')
    handles = [cooperation_patch, conflict_patch]
    if 'RollingMean' in daily_conflict.columns:
        rolling_line = mpatches.Patch(color='blue', label=f'{window_size}-day Moving Average')
        handles.append(rolling_line)
    plt.legend(handles=handles, fontsize=12)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'conflict_time_series.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved conflict time series plot to {output_path}")

def visualize_top_country_pairs(df, output_dir='visualizations', top_n=20):
    """
    Visualize the most frequent country pairs
    """
    if 'CountryPair' not in df.columns:
        print("Warning: CountryPair column not found")
        return
    
    # Get count of articles for each country pair
    country_counts = df['CountryPair'].value_counts().reset_index()
    country_counts.columns = ['CountryPair', 'Count']
    
    # Get top N pairs
    top_pairs = country_counts.head(top_n)
    
    plt.figure(figsize=(14, 10))
    
    # Create horizontal bar chart
    bars = plt.barh(top_pairs['CountryPair'], top_pairs['Count'], color='skyblue')
    
    # Add count labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{width:,.0f}', ha='left', va='center', fontsize=10)
    
    # Set labels and title
    plt.xlabel('Number of Articles', fontsize=14)
    plt.ylabel('Country Pair', fontsize=14)
    plt.title(f'Top {top_n} Most Frequent Country Pairs in Dataset', fontsize=16)
    
    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3)
    
    # Adjust layout to fit all country pair labels
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'top_country_pairs.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top country pairs plot to {output_path}")

def visualize_sentiment_by_country_pair(df, sentiment_col='SimpleSentiment', output_dir='visualizations', top_n=15):
    """
    Visualize average sentiment for top country pairs
    """
    if 'CountryPair' not in df.columns:
        print("Warning: CountryPair column not found")
        return
    
    # Calculate average sentiment and article count for each country pair
    pair_sentiment = df.groupby('CountryPair').agg({
        sentiment_col: 'mean',
        'EventID': 'count'
    }).reset_index()
    
    # Rename columns for clarity
    pair_sentiment.columns = ['CountryPair', 'AverageSentiment', 'ArticleCount']
    
    # Filter to pairs with at least 5 articles (for statistical significance)
    min_articles = 5
    pair_sentiment = pair_sentiment[pair_sentiment['ArticleCount'] >= min_articles]
    
    # Sort by article count and get top N
    top_pairs = pair_sentiment.sort_values('ArticleCount', ascending=False).head(top_n)
    
    # Sort by sentiment for visualization
    top_pairs = top_pairs.sort_values('AverageSentiment', ascending=True)
    
    plt.figure(figsize=(14, 10))
    
    # Create horizontal bar chart with color based on sentiment
    bars = plt.barh(top_pairs['CountryPair'], top_pairs['AverageSentiment'], 
                   color=plt.cm.RdYlGn(top_pairs['AverageSentiment'] / 2 + 0.5))  # Normalize to 0-1 range
    
    # Add sentiment value labels to the bars
    for bar in bars:
        width = bar.get_width()
        color = 'black' if abs(width) < 0.3 else 'white'
        plt.text(width + (0.02 if width >= 0 else -0.08), 
                bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left' if width >= 0 else 'right', 
                va='center', fontsize=10, color=color, fontweight='bold')
    
    # Add zero reference line
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Set labels and title
    plt.xlabel('Average Sentiment Score', fontsize=14)
    plt.ylabel('Country Pair', fontsize=14)
    plt.title(f'Average Sentiment for Top {top_n} Most Frequent Country Pairs', fontsize=16)
    
    # Add annotation for article count
    for i, (_, row) in enumerate(top_pairs.iterrows()):
        plt.text(plt.xlim()[0] + 0.05, i, f'n={row["ArticleCount"]}', 
                va='center', ha='left', fontsize=9, alpha=0.7)
    
    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3)
    
    # Adjust layout to fit all country pair labels
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'sentiment_by_country_pair.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sentiment by country pair plot to {output_path}")

def visualize_sentiment_calendar_heatmap(df, sentiment_col='SimpleSentiment', output_dir='visualizations', year=None):
    """
    Create a calendar heatmap of sentiment
    """
    if 'DATE' not in df.columns:
        print("Warning: DATE column not found")
        return
    
    # Ensure DATE is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['DATE']):
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Get available years in data
    years = sorted(df['DATE'].dt.year.unique())
    if not years:
        print("Warning: No valid years found in data")
        return
    
    # If year is not specified, use the first full year in the data
    if year is None:
        year = years[0]
    
    # Filter to the specified year
    year_df = df[df['DATE'].dt.year == year]
    if year_df.empty:
        print(f"Warning: No data found for year {year}")
        return
    
    # Group by date and calculate average sentiment
    daily_sentiment = year_df.groupby(year_df['DATE'].dt.date)[sentiment_col].mean().reset_index()
    daily_sentiment['DATE'] = pd.to_datetime(daily_sentiment['DATE'])
    
    # Create date range for the full year
    start_date = pd.Timestamp(f'{year}-01-01')
    end_date = pd.Timestamp(f'{year}-12-31')
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Create dataframe with all dates
    all_dates = pd.DataFrame({'DATE': date_range})
    all_dates['day'] = all_dates['DATE'].dt.day
    all_dates['month'] = all_dates['DATE'].dt.month
    all_dates['year'] = all_dates['DATE'].dt.year
    all_dates['dayofweek'] = all_dates['DATE'].dt.dayofweek  # Monday=0, Sunday=6
    
    # Merge with sentiment data
    all_dates = all_dates.merge(daily_sentiment, on='DATE', how='left')
    
    # Create figure with subplot for each month
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # Define colormap
    cmap = sns.diverging_palette(10, 133, as_cmap=True)  # Red to Blue
    
    # Calculate global min/max for consistent color scaling
    vmin = daily_sentiment[sentiment_col].min()
    vmax = daily_sentiment[sentiment_col].max()
    
    # For each month
    for month in range(1, 13):
        ax = axes[month-1]
        
        # Filter data for this month
        month_data = all_dates[all_dates['month'] == month]
        
        # Create calendar grid for the month
        cal_data = np.zeros((6, 7)) * np.nan  # Up to 6 weeks, 7 days each
        
        # Fill in the data
        for _, row in month_data.iterrows():
            # Calculate position
            week_of_month = (row['day'] - 1 + pd.Timestamp(f"{year}-{month}-01").dayofweek) // 7
            day_of_week = row['dayofweek']
            if week_of_month < 6:  # Ensure we don't go out of bounds
                cal_data[week_of_month, day_of_week] = row[sentiment_col]
        
        # Create heatmap
        sns.heatmap(cal_data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                   cbar=False, square=True, linewidths=1, linecolor='white',
                   mask=np.isnan(cal_data))
        
        # Set title and labels
        ax.set_title(calendar.month_name[month], fontsize=16)
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=10)
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Add color bar to figure
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Sentiment Score', fontsize=14)
    
    # Set figure title
    plt.suptitle(f'Sentiment Calendar Heatmap - {year}', fontsize=24, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, f'sentiment_calendar_{year}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sentiment calendar heatmap for {year} to {output_path}")

def create_all_visualizations(data_file="processed_worldrep.csv", output_dir="visualizations"):
    """
    Create all visualizations
    """
    print(f"Starting visualization process for {data_file}")
    
    # Load data
    df = load_processed_data(data_file)
    
    # Determine whether we're using VADER or simple sentiment
    sentiment_col = 'SentimentCompound' if 'SentimentCompound' in df.columns else 'SimpleSentiment'
    print(f"Using sentiment column: {sentiment_col}")
    
    # Create all visualizations
    visualize_sentiment_distribution(df, sentiment_col, output_dir)
    visualize_conflict_score_distribution(df, output_dir)
    visualize_sentiment_vs_conflict(df, sentiment_col, output_dir)
    visualize_sentiment_time_series(df, sentiment_col, output_dir)
    visualize_conflict_time_series(df, output_dir)
    visualize_top_country_pairs(df, output_dir)
    visualize_sentiment_by_country_pair(df, sentiment_col, output_dir)
    
    # Create calendar heatmap for each full year in the data
    years = sorted(df['DATE'].dt.year.unique()) if 'DATE' in df.columns else []
    for year in years:
        visualize_sentiment_calendar_heatmap(df, sentiment_col, output_dir, year)
    
    print("All visualizations completed!")

if __name__ == "__main__":
    # Run all visualizations
    create_all_visualizations()