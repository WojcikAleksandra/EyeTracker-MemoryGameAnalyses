"""
Example script for analyzing Memory Game eye tracking data.

This demonstrates basic analysis techniques for the game_data CSV files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def load_latest_game_data():
    """Load the most recent game data file."""
    files = glob.glob("game_data_*.csv")
    if not files:
        print("No game data files found.")
        return None
    
    latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
    print(f"Loading: {latest_file}")
    return pd.read_csv(latest_file)


def basic_statistics(df):
    """Print basic statistics about the game."""
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    # Game duration
    duration_ms = df['ms'].max()
    print(f"Game duration: {duration_ms/1000:.2f} seconds")
    
    # Valid gaze samples
    total_samples = len(df)
    valid_samples = len(df[df['valid'] == 1])
    print(f"Total gaze samples: {total_samples}")
    print(f"Valid gaze samples: {valid_samples} ({valid_samples/total_samples*100:.1f}%)")
    
    # Clicks
    clicks = df[df['flip'] != -1]
    total_clicks = len(clicks)
    matches = clicks[clicks['matched'] == 1]
    mismatches = clicks[clicks['matched'] == 0]
    
    print(f"\nTotal clicks: {total_clicks}")
    print(f"Matches: {len(matches)} pairs")
    print(f"Mismatches: {len(mismatches)} attempts")
    
    # Calculate moves (pairs of clicks)
    moves = total_clicks // 2
    print(f"Total moves: {moves}")
    
    return {
        'duration_s': duration_ms / 1000,
        'valid_rate': valid_samples / total_samples,
        'total_clicks': total_clicks,
        'matches': len(matches),
        'moves': moves
    }


def card_attention_analysis(df):
    """Analyze attention paid to each card."""
    print("\n" + "="*60)
    print("CARD ATTENTION ANALYSIS")
    print("="*60)
    
    # Get unique card IDs (excluding -1 which means no card)
    card_ids = sorted(df[df['card_id_gaze'] != -1]['card_id_gaze'].unique())
    
    if len(card_ids) == 0:
        print("No card attention data available.")
        return
    
    # Approximate sampling rate
    sampling_interval_ms = 16  # ~60 FPS
    
    attention_data = []
    for card_id in card_ids:
        frames = len(df[df['card_id_gaze'] == card_id])
        duration_s = frames * sampling_interval_ms / 1000
        attention_data.append({
            'card_id': card_id,
            'frames': frames,
            'duration_s': duration_s
        })
    
    attention_df = pd.DataFrame(attention_data)
    attention_df = attention_df.sort_values('duration_s', ascending=False)
    
    print("\nTime spent looking at each card:")
    print(attention_df.to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(attention_df['card_id'].astype(str), attention_df['duration_s'])
    plt.xlabel('Card ID')
    plt.ylabel('Time (seconds)')
    plt.title('Attention Duration per Card')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('attention_per_card.png', dpi=150)
    print("\nSaved: attention_per_card.png")
    
    return attention_df


def gaze_trajectory_plot(df):
    """Create a gaze trajectory visualization."""
    print("\n" + "="*60)
    print("GAZE TRAJECTORY PLOT")
    print("="*60)
    
    valid_gaze = df[df['valid'] == 1]
    
    if len(valid_gaze) == 0:
        print("No valid gaze data to plot.")
        return
    
    # Create scatter plot colored by time
    plt.figure(figsize=(14, 9))
    
    scatter = plt.scatter(
        valid_gaze['x_gaze'], 
        valid_gaze['y_gaze'],
        c=valid_gaze['ms'],
        cmap='viridis',
        alpha=0.4,
        s=20
    )
    
    # Add clicks as red X markers
    clicks = df[df['flip'] != -1]
    if len(clicks) > 0:
        plt.scatter(
            clicks['x_click'],
            clicks['y_click'],
            c='red',
            marker='x',
            s=200,
            linewidths=3,
            label='Clicks',
            zorder=10
        )
    
    plt.colorbar(scatter, label='Time (ms)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Gaze Trajectory During Game')
    plt.legend()
    plt.gca().invert_yaxis()  # Screen coordinates (0,0 at top-left)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('gaze_trajectory.png', dpi=150)
    print("Saved: gaze_trajectory.png")


def fixation_analysis(df, fixation_radius=50, min_duration_ms=100):
    """Simple fixation detection based on spatial clustering."""
    print("\n" + "="*60)
    print("FIXATION ANALYSIS")
    print("="*60)
    
    valid_gaze = df[df['valid'] == 1].copy()
    
    if len(valid_gaze) < 10:
        print("Not enough valid gaze data for fixation analysis.")
        return
    
    # Simple fixation detection: consecutive points within radius
    fixations = []
    current_fixation = []
    
    for idx, row in valid_gaze.iterrows():
        if len(current_fixation) == 0:
            current_fixation.append(row)
        else:
            # Check distance from fixation center
            fx = np.mean([r['x_gaze'] for r in current_fixation])
            fy = np.mean([r['y_gaze'] for r in current_fixation])
            dist = np.sqrt((row['x_gaze'] - fx)**2 + (row['y_gaze'] - fy)**2)
            
            if dist < fixation_radius:
                current_fixation.append(row)
            else:
                # End current fixation
                if len(current_fixation) > 0:
                    duration = current_fixation[-1]['ms'] - current_fixation[0]['ms']
                    if duration >= min_duration_ms:
                        fixations.append({
                            'x': fx,
                            'y': fy,
                            'duration_ms': duration,
                            'start_ms': current_fixation[0]['ms'],
                            'n_samples': len(current_fixation)
                        })
                current_fixation = [row]
    
    # Handle last fixation
    if len(current_fixation) > 0:
        duration = current_fixation[-1]['ms'] - current_fixation[0]['ms']
        if duration >= min_duration_ms:
            fx = np.mean([r['x_gaze'] for r in current_fixation])
            fy = np.mean([r['y_gaze'] for r in current_fixation])
            fixations.append({
                'x': fx,
                'y': fy,
                'duration_ms': duration,
                'start_ms': current_fixation[0]['ms'],
                'n_samples': len(current_fixation)
            })
    
    print(f"\nDetected {len(fixations)} fixations")
    print(f"Average fixation duration: {np.mean([f['duration_ms'] for f in fixations]):.0f} ms")
    print(f"Total fixation time: {sum([f['duration_ms'] for f in fixations])/1000:.2f} seconds")
    
    # Plot fixations
    if len(fixations) > 0:
        fix_df = pd.DataFrame(fixations)
        
        plt.figure(figsize=(14, 9))
        plt.scatter(
            fix_df['x'],
            fix_df['y'],
            s=fix_df['duration_ms']/2,  # Size proportional to duration
            c=fix_df['start_ms'],
            cmap='coolwarm',
            alpha=0.6,
            edgecolors='black',
            linewidths=1
        )
        plt.colorbar(label='Time (ms)')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Fixation Map (size = duration)')
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('fixation_map.png', dpi=150)
        print("Saved: fixation_map.png")
    
    return fixations


def click_prediction_analysis(df):
    """Analyze relationship between gaze and clicks."""
    print("\n" + "="*60)
    print("CLICK PREDICTION ANALYSIS")
    print("="*60)
    
    clicks = df[df['flip'] != -1].copy()
    
    if len(clicks) == 0:
        print("No click data available.")
        return
    
    # For each click, look at gaze 500ms before
    results = []
    
    for idx, click in clicks.iterrows():
        click_time = click['ms']
        pre_click_window = df[
            (df['ms'] >= click_time - 500) & 
            (df['ms'] < click_time) &
            (df['valid'] == 1)
        ]
        
        if len(pre_click_window) > 0:
            # Check if user was looking at the card they clicked
            looking_at_card = pre_click_window[
                pre_click_window['card_id_gaze'] == click['card_id_click']
            ]
            
            fraction = len(looking_at_card) / len(pre_click_window)
            
            results.append({
                'click_time_ms': click_time,
                'card_id': click['card_id_click'],
                'gaze_on_card_fraction': fraction,
                'matched': click['matched']
            })
    
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        avg_fraction = results_df['gaze_on_card_fraction'].mean()
        
        print(f"\n500ms before click analysis:")
        print(f"Average time looking at clicked card: {avg_fraction*100:.1f}%")
        print(f"\nBreakdown:")
        print(f"  Matched pairs: {results_df[results_df['matched']==1]['gaze_on_card_fraction'].mean()*100:.1f}%")
        print(f"  Mismatched pairs: {results_df[results_df['matched']==0]['gaze_on_card_fraction'].mean()*100:.1f}%")
        
        return results_df
    else:
        print("Not enough data for analysis.")
        return None


def main():
    """Run all analyses on the latest game data."""
    print("="*60)
    print("MEMORY GAME EYE TRACKING DATA ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_latest_game_data()
    if df is None:
        print("\nNo data to analyze. Please play a game first!")
        return
    
    # Run analyses
    stats = basic_statistics(df)
    attention = card_attention_analysis(df)
    gaze_trajectory_plot(df)
    fixations = fixation_analysis(df)
    click_pred = click_prediction_analysis(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated plots:")
    print("  - attention_per_card.png")
    print("  - gaze_trajectory.png")
    print("  - fixation_map.png")
    print("\nYou can now examine these visualizations!")


if __name__ == "__main__":
    main()

