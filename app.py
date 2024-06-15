import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_icon='soccer', layout='wide')

# Load your datasets
player_df = pd.read_csv("./2021-2022 Football Player Stats.csv", encoding='ISO-8859-1', delimiter=';')
team_df = pd.read_csv("./2021-2022 Football Team Stats.csv", encoding='ISO-8859-1', delimiter=';')
player_df.dropna(inplace=True)

# Merged Lg Rk column to player df 
df = player_df.merge(team_df[['Squad', 'LgRk']], on='Squad', how='left')

# Convert 'LgRk' to numeric values
df['LgRk'] = pd.to_numeric(df['LgRk'], errors='coerce')

# Preprocessing to filter players with minimum matches played
df = df[df['MP'] > 8]

# Identify numerical features and exclude specific columns (numeric but irrelevant)
exclusions = ['LgRk', 'Rk', 'Born', 'MP', 'Starts', 'Min', '90s', 'Age']
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(exclusions)

# Slopes here indicates the linear relationship between Lg Rk and the features
positive_features = []
negative_features = []

# Taking Mean values wrt Individual Squad for better results

for feature in numerical_features:
    # Calculate the mean of the feature for each squad
    mean_features = df.groupby('Squad')[feature].apply('mean').reset_index()
    
    # Merge to get 'LgRk' for each squad
    mean_features_with_lgrk = mean_features.merge(df[['Squad', 'LgRk']].drop_duplicates(), on='Squad', how='left')
    
    # Calculate the best-fit line
    m, b = np.polyfit(mean_features_with_lgrk['LgRk'], mean_features_with_lgrk[feature], 1)
    
    # Classify the slopes
    if m > 0.013:
        positive_features.append(feature)
    elif m < 0:
        negative_features.append(feature)
    
    # # Create x values for the best-fit line plot
    # x_values = np.array(mean_features_with_lgrk['LgRk'])
    
    # # Calculate y values based on the slope (m), intercept (b), and x_values
    # y_values = m * x_values + b

    # # Plotting is optional here, based on your need to visualize
    # plt.figure(figsize=(10, 4))
    # plt.scatter(mean_features_with_lgrk['LgRk'], mean_features_with_lgrk[feature], alpha=0.5, label='Data Points')
    # plt.plot(x_values, y_values, 'r-', label='Best Fit Line')
    # plt.title(f"{feature} vs. League Rank")
    # plt.xlabel('League Rank (LgRk)')
    # plt.ylabel(f"Average {feature}")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
            

            
# from collections import Counter
# # Create a Counter object
# item_counts1 = Counter(positive_slopes)
# item_counts2 = Counter(negative_slopes)

# # Find features that appear more than once in the mean and median slope values
# positive_features = [item for item, count in item_counts1.items() if count > 1]
# negative_features = [item for item, count in item_counts2.items() if count > 1]

# print("Positively related features to LgRk v: \n",positive_features)
# print(len(positive_features))
# print()
# print("Negatively related features to LgRk v: \n",negative_features)
# print(len(negative_features))

# Function to calculate squad weaknesses based on numerical features
def calculate_squad_weaknesses(squad_name):
    """
    Returns the Squad's Weak Features.
    """
    
    # Combined list of features
    features = positive_features + negative_features

    # Store the squad's weak feature name and its relative rank
    squad_weaknesses = {}

    # Determine the competition of the squad
    squad_comp = df[df['Squad'] == squad_name]['Comp'].iloc[0]

    for feature in features:
        # Decide the direction of ranking based on feature nature
        is_feature_positive = feature in positive_features
        ascending_rank = is_feature_positive  # Ascend if feature is positive, descend if negetive
        
        # Rank squads in their competition by feature
        squad_feature_rank = df[df['Comp'] == squad_comp].groupby(['Squad'])[feature].mean().rank(method='dense', ascending=ascending_rank)
        squad_feature_rank = squad_feature_rank.reset_index(name=f'{feature}Rank')
        
        # Find the rank of the specified squad in its competition
        squad_rank = squad_feature_rank.loc[squad_feature_rank['Squad'] == squad_name, f'{feature}Rank'].values[0]
        
        # Calculate the total number of squads in the squad's competition
        total_squads = df[df['Comp'] == squad_comp]['Squad'].nunique()
        
        # Identify if the feature is a weakness (if it is in the relative bottom 5)
        if squad_rank > total_squads - 5:
            squad_weaknesses[feature] = squad_rank
       
    return squad_weaknesses


# Function to find top players to address squad weaknesses
def recommend_players_for_squad(squad_name):
    """
    Recommends Top 5 fit player profiles based on the squad's weaknessess.
    """
    squad_weaknesses = calculate_squad_weaknesses(squad_name)
    weak_features = list(squad_weaknesses.keys())

    # Calculate player ranks in weak areas, adjusting for feature nature
    player_ranks = df[weak_features + ['Comp', 'Player', 'Pos', 'Squad', 'LgRk']].copy()  
    for feature in weak_features:
        # Determine if the current feature is positive or negative
        is_feature_positive = feature in positive_features
        # Rank players appropriately: ascending=True for postive features, ascending=False for negative.
        ascending_rank = is_feature_positive

        # Player Ranks for weakness features of the squad
        player_ranks[f'{feature}Rank'] = player_ranks.groupby('Comp')[feature].rank(method='dense', ascending=ascending_rank)
    
    # Calculate average rank across weak features for each player
    rank_columns = [f'{feature}Rank' for feature in weak_features]
    player_ranks['FinalRank'] = player_ranks[rank_columns].mean(axis=1)
    
    # Filter top 5 players based on average rank
    # Since FinalRank is an average of ranks, a lower FinalRank indicates a player who consistently ranks well.
    top_players = player_ranks.nsmallest(5, 'FinalRank')[['Player','Pos', 'Squad', 'FinalRank']]  
    top_players['FinalRank'] = top_players['FinalRank'].rank()
    
    return top_players


# # Example usage
# squad_name = 'Manchester Utd'
# top_recommendations = recommend_players_for_squad(squad_name)
# print(f"Team : {squad_name}\nWeakness : {pd.Series(calculate_squad_weaknesses(squad_name).keys()).tolist()}")
# top_recommendations.reset_index(drop=True)
def main():
    st.title("Football Player Recommender")
    st.text("""
            This Simplified Analysis uses player and team statistics (player_df and team_df) of 2021-2022 season only
            to assess performance metrics like goals and assists against league rankings (LgRk).
            By identifying correlations and peer-relative squad weaknesses, it recommends players who excel in critical areas, 
            aiding in puerly data-driven strategic recruitment for improved team performance with suitable enhancments in future
            in competitive football leagues.
        """)
    
    squad_name = st.selectbox("Select any club:" ,options=team_df['Squad'].unique().tolist())
    if squad_name:
        top_recommendations = recommend_players_for_squad(squad_name)
        st.subheader(f"Weak Features :\n{pd.Series(calculate_squad_weaknesses(squad_name).keys()).tolist()}")
        st.subheader(f"Best Fit Player Profiles for {squad_name}:")
        return st.dataframe(top_recommendations.set_index('FinalRank'))

if __name__ == '__main__':
    main()