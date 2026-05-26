# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %%
# # International Soccer
#
# ## GSAC workshop 3 (25/26)
#
# ## TMP outline
# 1. Setup (10 min) — load CSV, .head(), .info(), .describe().
# 2. Cleaning (15 min) — parse dates, derive winner, goal_diff, is_home.
# 3. Guided questions (45 min:
#
#     - Which country has the best all-time win rate (min 100 matches)?
#     - Does home advantage exist? Quantify it.
#     - How have goals per match changed since 1950?
#     - World Cup–only subset: who overperforms vs friendlies?
#
# 4. Viz (20 min) — matplotlib/seaborn: line chart of goals/match over time, bar chart of top teams, heatmap of head-to-head.
# 5. ML stretch goal (30 min) — pivot to FIFA players CSV: predict overall from a handful of attributes with LinearRegression, then RandomForestRegressor; compare R² and feature importances.
#
# ## ideas
# - predict world cup winner
#
# ## interesting findings
# - San Marino Sucks
# - Some things shouldn't have national teams
# - 7-1 was insane
# - Some teams have only played home or awway

# %%
# ## Getting Started
# Let's download the dataset and import the libraries we'll be using.

# %%
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import poisson, skellam
import seaborn as sns
path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")

# %%
# ## Our Data
# Let's take a look at our data. The first dataset we'll be analyzing (`results.csv`) contains game data from all international soccer matches, dating back to 1872!
#
# Here are some useful functions to quickly get a feel for our dataset:
#   - `.head(n)`, `.tail(n)`: show the frist/last `n` entries in the dataframe
#   - `.info()`: show the columns and their types
#   - `.describe()`: get statistical data for each *numeric* column (eg. mean, min, max, etc.)
#   - `.nunique()`: show how many unique values each column has
#   - `[column]`: get a specific column

# %%
# Load the dataset into a DataFrame object
results = pd.read_csv(os.path.join(path, "results.csv"))
results = results.dropna() # remove unplayed matches

# Show the first 5 entries
results.head()

# %%

# TODO
# use some of the funcitons above to get a better understanding of our dataset
...

# %%
# ### Let's start adding to the dataset
# We'll start by doing some simple calculations: goal difference, outcome, etc

# %%
# We create the `year` column, we'll use this later!
results['year'] = results['date'].apply(lambda d: np.int64(d[0:4]))

# You can treat columns like numbers, doing methematical operations and applying functions...
results['goal_difference']     = results['home_score'] - results['away_score']
results['goal_difference_abs'] = results['goal_difference'].abs()

# TODO: create the `total_goals` column (the some of goals scored both teams)
# results['total_goals'] = ...
results['total_goals'] = results['home_score'] + results['away_score']

# Here, we apply a function to every row to get the winner/loser/draw
def get_winner(row):
  if row['goal_difference'] == 0:
    return 'draw'
  elif row['goal_difference'] > 0:
    return row['home_team']
  else: # row['goal_difference'] < 0:
    return row['away_team']
def get_loser(row):
  if row['goal_difference'] == 0:
    return 'draw'
  elif row['goal_difference'] < 0:
    return row['home_team']
  else: # row['goal_difference'] > 0:
    return row['away_team']
results['winner'] = results.apply(get_winner, axis=1)
results['loser']  = results.apply(get_loser,  axis=1)

results.head()

# %%
# Here, we calculate the total games played per team, and add it to the dataframe

# %%
# `datframe[column].value_counts()` gets amount of time each value occurs in the column
home_games_played = results['home_team'].value_counts()
away_games_played = results['away_team'].value_counts()

# we each teams' total games played by combining home and away games played
games_played = pd.concat([home_games_played, away_games_played]).groupby(level=0).sum()

# now we add it back into the results
results['home_team_games_played'] = results['home_team'].apply(lambda t: games_played[t])
results['away_team_games_played'] = results['away_team'].apply(lambda t: games_played[t])
results.head()

# %%
# ## Let's dig!
# Now that we have built uppon our original dataset, let's try and find some interesing facts
#
# We'll use the dataframe's filtering, grouping, and sorting functionality to do so:

# %%
# We can FILTER a daset
# Again, we can treat columns like values, comparing them to a refence
wc_results = results[results['tournament'] == "FIFA World Cup"] # get all games played in world cups

# We can combine filters with | (or), & (and), and ~(not)
wc_results = wc_results[                                        # get all games where a team scored 7 or more goals
    (wc_results['home_score'] >= 7) |
    (wc_results['away_score'] >= 7)
]

# Now, can can get the losers in these matches, and see how many times it happened
wc_results['loser'].value_counts()

# TODO:
# In World Cup matches, Brazil has only conceded 7+ goals in a loss once (7-1).
# How many times has Germany won by 7+ goals in World Cup matches?
##### wc_results['winer'].value_counts() # find germany

# %%
# ### Team Stats
#
# For the next part, we'll look at stats team-by-team.
# We've created a function to get stats by team. Feel free to look at it if you want, but here is a quick break down of some uses. **Run the cell to load the function**
#
# #### Get all team stats
# ```
# team_stats = get_team_stats()
# ```
#
# #### Get One Team's stats
# ```
# ireland_stats = get_team_stats(teams='Ireland')
# ```
#
# #### Get stats for a specific tournament
# ```
# euro_stats = get_team_stats(tournament='UEFA Euro')
# ```
#
# #### Get stats for a specific matchup
# ```
# us_vs_mx_stats = get_team_stats(matchup=['United States', 'Mexico'])
# ```

# %%
def get_team_stats(tournament=None, sided_only=False, neutral_only=False, matchup=None, teams=None, date_min=None, date_max=None):
  data = results.dropna()
  if tournament:
    data = data[data['tournament'] == tournament]
  if sided_only:
    data = data[data['neutral'] == False]
  elif neutral_only:
    data = data[data['neutral'] == True]
  if date_min:
    data = data[data['date'] > date_min]
  if date_max:
    data = data[data['date'] < date_max]
  if matchup:
    data['matchup'] = data.apply(lambda row: ','.join(sorted([row['home_team'], row['away_team']])), axis=1)
    matchup = ','.join(sorted([matchup[0], matchup[1]]))
    data = data[data['matchup'] == matchup]
  if teams:
    if isinstance(teams, str):
      teams = [teams]
    data = data[(data['home_team'].isin(teams)) | (data['away_team'].isin(teams))]

  hgp = data['home_team'].value_counts()
  agp = data['away_team'].value_counts()
  gp = pd.concat([hgp, agp]).groupby(level=0).sum()
  # teams = gp.index

  stats = pd.DataFrame(gp).sort_index().rename(columns={"count": "games_played"})

  stats['home_games'] = hgp
  stats['home_games'] = stats['home_games'].fillna(0).astype(np.int64)
  stats['away_games'] = agp
  stats['away_games'] = stats['away_games'].fillna(0).astype(np.int64)

  stats['home_wins'] = data[data["goal_difference"] > 0].groupby('winner').size()
  stats['home_wins'] = stats['home_wins'].fillna(0).astype(np.int64)

  stats['home_losses'] = data[data["goal_difference"] < 0].groupby('loser').size()
  stats['home_losses'] = stats['home_losses'].fillna(0).astype(np.int64)

  stats['home_draws'] = data[data["goal_difference"] == 0].groupby('home_team').size()
  stats['home_draws'] = stats['home_draws'].fillna(0).astype(np.int64)

  stats['away_wins'] = data[data["goal_difference"] < 0].groupby('winner').size()
  stats['away_wins'] = stats['away_wins'].fillna(0).astype(np.int64)

  stats['away_losses'] = data[data["goal_difference"] > 0].groupby('loser').size()
  stats['away_losses'] = stats['away_losses'].fillna(0).astype(np.int64)

  stats['away_draws'] = data[data["goal_difference"] == 0].groupby('away_team').size()
  stats['away_draws'] = stats['away_draws'].fillna(0).astype(np.int64)

  stats.insert(1, "wins", stats['away_wins'] + stats['home_wins'])
  stats.insert(2, "draws", stats['away_draws'] + stats['home_draws'])
  stats.insert(3, "losses", stats['away_losses'] + stats['home_losses'])

  stats['win_percentage'] = stats['wins'] / stats['games_played']
  stats['draw_percentage'] = stats['draws'] / stats['games_played']
  stats['loss_percentage'] = stats['losses'] / stats['games_played']

  stats['home_win_percentage'] = stats['home_wins'] / stats['home_games']
  stats['home_draw_percentage'] = stats['home_draws'] / stats['home_games']
  stats['home_loss_percentage'] = stats['home_losses'] / stats['home_games']

  stats['away_win_percentage'] = stats['away_wins'] / stats['away_games']
  stats['away_draw_percentage'] = stats['away_draws'] / stats['away_games']
  stats['away_loss_percentage'] = stats['away_losses'] / stats['away_games']

  stats['home_away_win_difference'] = stats['home_win_percentage'] - stats['away_win_percentage']

  if teams:
    return stats.loc[teams]
  return stats

# get_team_stats(tournament='FIFA World Cup', matchup=['Brazil', 'Germany']).sort_values('games_played', ascending=False)
# results[(results['home_team'] == 'Sweden') | (results['away_team'] == 'Sweden')]
# results[
    # (((results['home_team'] == 'Brazil') & (results['away_team'] == 'Germany')) |
    # ((results['home_team'] == 'Germany') & (results['away_team'] == 'Brazil'))) &
    # (results['tournament'] == 'FIFA World Cup')
# ]


# %%
# ## Winers ... and Losers
#
# Lets look at some of the most wining and losing teams.
#
# There are a lot of columns here, but the ones we care about the most are `games_played`, `wins`, and `win_percentage`

# %%
# Get per-team stats
team_stats = get_team_stats()
team_stats.head()

# %%
# Lets find the team with the best wining percentage, having played at least 100 games

# %%
# Filter out teams with less that 100 games played
team_stats_played100 = team_stats[team_stats['games_played'] >= 100]

# Here we are keeping only the `win_percentage`, `games_played` columns
team_stats_played100 = team_stats_played100[['win_percentage', 'games_played']]

# Use the .sort() function to sort by `win_percentage` from high to low
team_stats_played100.sort_values('win_percentage', ascending=False).head(5)

# %%
# Now lets find out who underpreforms at world cups by comparing world cup win percentage to friendly win percentage

# %%
# TODO: get start for only FIFA World Cup games
# wc_team_stats = ...

# # this is a list of teams that have played in a world cup
# wc_teams = wc_team_stats.index

# # TODO: get start for only friendlies (hint: the dataset the tournament type is 'Friendly')
# friendly_stats_all_teams = ...

# # Get the stats from friendlies of teams that have played in a world cup
# friendly_stats_wc_teams = friendly_stats_all_teams[friendly_stats_all_teams.index.isin(wc_teams)]

# # We ceate a new data frame with each team and their win percentage in friendlies and the world cup
# # The data frame was columns `wc_win_percentage` and `f_win_percentage`
# win_percetages = pd.DataFrame({
#     "wc_win_percentage": wc_team_stats['win_percentage'],
#     "f_win_percentage": friendly_stats_wc_teams['win_percentage']
# })

# # TODO: find the difference bewteen world cup win percentage and friendly win percentage
# win_percetages["difference"] = ...

# # TODO: display the top 5 teams by win percentage difference
# ...



#### ANSWER ####
# TODO: get start for only FIFA World Cup games
wc_team_stats = get_team_stats(tournament='FIFA World Cup')

# this is a list of teams that have played in a world cup
wc_teams = wc_team_stats.index

# TODO: get start for only friendlies (hint: the dataset the tournament type is 'Friendly')
friendly_stats_all_teams = get_team_stats(tournament='Friendly')

# Get the stats from friendlies of teams that have played in a world cup
friendly_stats_wc_teams = friendly_stats_all_teams[friendly_stats_all_teams.index.isin(wc_teams)]

# We ceate a new data frame with each team and their win percentage in friendlies and the world cup
# The data frame was columns `wc_win_percentage` and `f_win_percentage`
win_percetages = pd.DataFrame({
    "wc_win_percentage": wc_team_stats['win_percentage'],
    "f_win_percentage": friendly_stats_wc_teams['win_percentage']
})

# TODO: find the difference bewteen world cup win percentage and friendly win percentage
win_percetages["difference"] = win_percetages['wc_win_percentage'] - win_percetages['f_win_percentage']

# TODO: display the bottom 5 teams by win percentage difference
win_percetages.sort_values('difference').head(5)

# %%
# ## Home Field Advantage
#
# It's lierally their country... Let's try and find out if home team advantage exists in internation soccer, and quantify it if possible

# %%
# Filter out results played on neutral fields
# Bonus: try yes neutral -> is there still home team advantage?
non_neutral = results[results['neutral'] == False]

# Calculate the amount of home and away wins
home_wins = len(non_neutral[non_neutral['goal_difference'] > 0])
away_wins = len(non_neutral[non_neutral['goal_difference'] < 0])

# Find the ratio of home wins to away wins
# If it is greater than 1, it means the home team is winning more than the away team
print("home/away win ratio: ", home_wins / away_wins)

# %%
# ## Poisson
# ### Explanation
# A distribution showing how many goals are expected to be scored, given an average amount of goals scored (eg. avg home goals = 1.79, avg away goals = 1.11).
#
# Note that the left-most bar shows when no goals are scored
#
# # TODO
# why might it be a good model
#
# ### Takeaways
#  - Away teams are much more likely to *not* score
#  - Home teams are more likely to score multiple goals
#  - Home teams are scoreless *less* than predicted, and Aaway teams are scoreless *more* than predicted

# %%
# https://dashee87.github.io/data%20science/python/home-advantage-in-football-leagues-around-the-world/
# https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling/
# https://github.com/dashee87/blogScripts/blob/master/Jupyter/2017-06-04-predicting-football-results-with-statistical-modelling.ipynb

# new dataframe 'home/away'
ha = results.dropna()
ha = ha[ha['neutral'] == False]
ha = ha[['home_score', 'away_score']]

# plot the acutally data
plt.hist(ha.values, bins=range(9), alpha=0.7, label=['Home', 'Away'], density=True, color=["#FFA07A", "#20B2AA"])

# plot the poisson prediction based the probability mass function
poisson_pred = np.column_stack([[poisson.pmf(i, ha.mean()[j]) for i in range(8)] for j in ha.mean().index])
plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,0],
                  linestyle='-', marker='o',label="Home", color = '#CD5C5C')
plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,1],
                  linestyle='-', marker='o',label="Away", color = '#006400')

# fancify the plot
legend = plt.legend(loc='upper right', fontsize=13, ncol=2)
legend.set_title("      Actual          Poisson        ", prop = {'size':'14', 'weight':'bold'})
plt.xticks([i-0.5 for i in range(1, 9)],[i for i in range(8)])
plt.xlabel("Goals per Match",size=13)
plt.ylabel("Proportion of Matches",size=13)
plt.title("Number of Goals per Match",size=14,fontweight='bold')
plt.show()


# %%
# # Skellam
# ## Explanation
# Is like a bell curve, but for the difference of two independant variables (eg. home goals - away goals). We are assuming they are independant
#
# # TODO
# why might it be a good model
#
# ## Takeaways
# - The draw (0) is the most common goal difference
# - Graph is slighty shifted to +gd, indicating
#

# %%
# plot the actual data
plt.hist(ha[['home_score']].values - ha[['away_score']].values, bins=range(-8,8), alpha=0.7, label='Actual', density=True, edgecolor='black')

# plot the skellham predection
skellam_pred = [skellam.pmf(i,  ha.mean()['home_score'],  ha.mean()['away_score']) for i in range(-7,8)]
plt.plot([i+0.5 for i in range(-7,8)], skellam_pred, linestyle='-', marker='o',label="Skellam", color = '#CD5C5C')

# make it pretty
plt.legend(loc='upper right', fontsize=13)
plt.xticks([i+0.5 for i in range(-7,8)],[i for i in range(-7,8)])
plt.xlabel("Goal Difference",size=13)
plt.ylabel("Proportion of Matches",size=13)
plt.title("Difference in Goals Scored (Home Team vs Away Team)",size=14,fontweight='bold')
plt.ylim([-0.004, 0.26])
plt.tight_layout()
plt.show()

# %%
# # Some Graphs

# %%
# ## Goals per game across time

# %%
results.groupby('year')['total_goals'].mean().plot()

# %%
top10 = team_stats.sort_values('wins', ascending=False).head(10)
top10_results = results[results['home_team'].isin(top10.index) | results['away_team'].isin(top10.index)]
top10_results

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create an empty list to store data for the new DataFrame
goal_diff_data = []

# Iterate through each row in top10_results
for index, row in top10_results.iterrows():
    # Add data for the home team if it's in the top 10
    if row['home_team'] in top10.index:
        goal_diff_data.append({
            'Team': row['home_team'],
            'Goal Difference': row['goal_difference']
        })
    # Add data for the away team if it's in the top 10
    if row['away_team'] in top10.index:
        goal_diff_data.append({
            'Team': row['away_team'],
            'Goal Difference': -row['goal_difference']  # Goal difference from away team's perspective
        })

# Create the DataFrame
goal_diff_df = pd.DataFrame(goal_diff_data)

# Create the boxplot
plt.figure(figsize=(15, 8))
sns.boxplot(x='Team', y='Goal Difference', data=goal_diff_df, order=top10.index) # Order by top10 wins for consistency
plt.title('Goal Difference Distribution for Top 10 Teams With Most Wins', fontsize=16)
plt.xlabel('Team', fontsize=12)
plt.ylabel('Goal Difference', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# TODO: do plots for a specific matchup (argentina vs uruguay most common, I think)

# %%
matchups = results.apply(lambda row: ' - '.join(sorted([row['home_team'], row['away_team']])), axis=1)
matchup_counts = matchups.value_counts().sort_index()
matchup_results = pd.DataFrame(matchup_counts)

results['matchup'] = matchups
matchup_results['total_goals_scored'] = results.groupby('matchup')['total_goals'].sum().sort_index()
matchup_results.sort_values('count', ascending=False)

# %%
# ## Visualizations for a Specific Matchup: Argentina vs. Uruguay

# %%
# Select a specific matchup, for example, Argentina vs. Uruguay (the most common one)
team1 = 'Argentina'
team2 = 'Uruguay'

# Filter for matches between these two teams
# We need to consider both home_team vs away_team and away_team vs home_team scenarios
matchup_df = results[
    (((results['home_team'] == team1) & (results['away_team'] == team2)) |
    ((results['home_team'] == team2) & (results['away_team'] == team1))) &
    (results['tournament'] == 'Copa América')
].dropna(subset=['home_score'])

# Count wins for each team and draws
team1_wins = matchup_df[matchup_df['winner'] == team1].shape[0]
team2_wins = matchup_df[matchup_df['winner'] == team2].shape[0]
draws = matchup_df[matchup_df['winner'] == 'draw'].shape[0]

# Create a DataFrame for plotting
matchup_summary = pd.DataFrame({
    'Outcome': [f'{team1} Wins', 'Draws', f'{team2} Wins'],
    'Count': [team1_wins, draws, team2_wins]
})

# Create the bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='Outcome', y='Count', data=matchup_summary, hue='Outcome', legend=False, palette=['skyblue', 'lightgray', 'lightcoral'])
plt.title(f'Head-to-Head Results: {team1} vs {team2}', fontsize=16)
plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Number of Matches', fontsize=12)
plt.show()

# %%
# This bar chart gives a quick overview of the historical performance between the two teams. We can also look at the goal difference distribution for this matchup to understand the competitiveness of their games.

# %%
# Prepare data for goal difference boxplot for the specific matchup
matchup_goal_data = []

for index, row in matchup_df.iterrows():
    if row['home_team'] == team1:
        matchup_goal_data.append({
            'Team': team1,
            'Goals': row['home_score']
        })
        matchup_goal_data.append({
            'Team': team2,
            'Goals': row['away_score']
        })
    else: # home_team is team2
        matchup_goal_data.append({
            'Team': team2,
            'Goals': row['home_score']
        })
        matchup_goal_data.append({
            'Team': team1,
            'Goals': row['away_score']
        })

matchup_goal_diff_df = pd.DataFrame(matchup_goal_data)

# Create the boxplot
# plt.figure(figsize=(10, 7))
sns.boxplot(x='Team', y='Goals', data=matchup_goal_diff_df, order=[team1, team2])
plt.title(f'Goal Scoring Distribution: {team1} vs {team2}', fontsize=16)
plt.xlabel('Team', fontsize=12)
plt.ylabel('Goals Scored', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
plt.show()

# %%
goalscorers = pd.read_csv(os.path.join(path, "goalscorers.csv"))
goalscorers[
    (((goalscorers['home_team'] == team1) & (goalscorers['away_team'] == team2)) |
    ((goalscorers['home_team'] == team2) & (goalscorers['away_team'] == team1)))
]
get_team_stats("Copa América", 'Argentina', 'Uruguay')

# %%
# TODO (maybe) try to predic 2026 wc

# %%

# %%
# !gdown 'https://drive.google.com/uc?id=1rGxbEL0qfGJGCgUeM9ze7iIam9PepPIO'

# %%
# rankings = pd.read_csv('~/Downloads/fifarankings.csv')
# rankings = rankings.drop(columns=['Rank', 'Last result', 'More'])
# rankings.insert(1, 'Rank', rankings.index + 1)
# rankings["Team"] = rankings["Team"].str.replace("IR Iran", "Iran").str.replace("Korea Republic", "South Korea").str.replace("USA", "United States").str.replace("Türkiye", "Turkey").str.replace("Czechia", "Czech Republic").str.replace("Côte d'Ivoire", "Ivory Coast").str.replace("Congo DR", "DR Congo").str.replace("Cabo Verde", "Cape Verde")

# rank_map = rankings.set_index('Team')

# display(rank_map.head())
# rank_map.to_csv("fifa_rankings")
rank_map = pd.read_csv("fifa_rankings.csv", index_col=0)
rank_map


# %%
recent_results = results.dropna()
recent_results = recent_results[recent_results['date'] > '2022-12-18'].reset_index(drop=True)
recent_results

# %%
wc_teams = {
    'Mexico': {
        'group': 'A'
    },
    'South Africa': {
        'group': 'A'
    },
    'South Korea': {
        'group': 'A'
    },
    'Czech Republic': {
        'group': 'A'
    },

    'Canada': {
        'group': 'B'
    },
    'Bosnia and Herzegovina': {
        'group': 'B'
    },
    'Qatar': {
        'group': 'B'
    },
    'Switzerland': {
        'group': 'B'
    },

    'Brazil': {
        'group': 'C'
    },
    'Morocco': {
        'group': 'C'
    },
    'Haiti': {
        'group': 'C'
    },
    'Scotland': {
        'group': 'C'
    },

    'United States': {
        'group': 'D'
    },
    'Paraguay': {
        'group': 'D'
    },
    'Australia': {
        'group': 'D'
    },
    'Turkey': {
        'group': 'D'
    },

    'Germany': {
        'group': 'E'
    },
    'Curaçao': {
        'group': 'E'
    },
    'Ivory Coast': {
        'group': 'E'
    },
    'Ecuador': {
        'group': 'E'
    },

    'Netherlands': {
        'group': 'F'
    },
    'Japan': {
        'group': 'F'
    },
    'Sweden': {
        'group': 'F'
    },
    'Tunisia': {
        'group': 'F'
    },

    'Belgium': {
        'group': 'G'
    },
    'Egypt': {
        'group': 'G'
    },
    'Iran': {
        'group': 'G'
    },
    'New Zealand': {
        'group': 'G'
    },

    'Spain': {
        'group': 'H'
    },
    'Cape Verde': {
        'group': 'H'
    },
    'Saudi Arabia': {
        'group': 'H'
    },
    'Uruguay': {
        'group': 'H'
    },

    'France': {
        'group': 'I'
    },
    'Senegal': {
        'group': 'I'
    },
    'Iraq': {
        'group': 'I'
    },
    'Norway': {
        'group': 'I'
    },

    'Argentina': {
        'group': 'J'
    },
    'Algeria': {
        'group': 'J'
    },
    'Austria': {
        'group': 'J'
    },
    'Jordan': {
        'group': 'J'
    },

    'Portugal': {
        'group': 'K'
    },
    'DR Congo': {
        'group': 'K'
    },
    'Uzbekistan': {
        'group': 'K'
    },
    'Colombia': {
        'group': 'K'
    },

    'England': {
        'group': 'L'
    },
    'Croatia': {
        'group': 'L'
    },
    'Ghana': {
        'group': 'L'
    },
    'Panama': {
        'group': 'L'
    }
}

import json

# groups = {}
# for g in "ABCDEFGHIJKL":
  # groups[g] = []
# for team in wc_teams:
  # groups[wc_teams[team]['group']].append(team)
# with open("groups.json", "w") as f:
# json.dump(groups, f)
with open("groups.json") as f:
  groups = json.load(f)
display(groups)
# wc_teams['Mexico']
# display(groups)
# list(wc_teams.keys())
# wc = get_team_stats(teams=list(wc_teams.keys()))
wc = pd.DataFrame.from_dict(wc_teams, orient='index')
wc.insert(1, 'points', np.zeros(48, dtype=np.int64))
# wc = wc[['group', 'points']]
wc['gd'] = np.zeros(48, dtype=np.int64)
wc['gf'] = np.zeros(48, dtype=np.int64)
wc['rank'] = -wc.index.map(lambda country: rank_map.loc[country]['rank'])
# help(wc.index.map)
# wc.to_csv("group_stage_leader_board.csv")
pd.read_csv("group_stage_leader_board.csv", index_col=0)


# %%
# pd.set_option('display.max_columns', None)
# outcomes3rd = pd.read_csv("~/Downloads/3rd.csv")
# outcomes3rd[outcomes3rd.colums[1:3]]
# display(outcomes3rd)
# better3rd = outcomes3rd[outcomes3rd.columns[14:]].set_index(
  # outcomes3rd[outcomes3rd.columns[1:13]].sum(axis=1)
# )
# better3rd.to_csv('3rd_place_outcomes.csv')
# o3 = pd.read_csv("3rd_place_outcomes.csv", index_col=0)
# with open("3rd_place_outcomes.json", "w") as f:
  # json.dump(dict(o3.to_json(orient='index')), f)
# o3.to_json("3rd_place_outcomes.json", orient='index')
with open("3rd_place_outcomes.json") as f:
  o3 = json.load(f)
o3

# %%
group_stage_matches = [
    [('A1', 'A2'), None],
    [('A3', 'A4'), None],
    [('A4', 'A2'), None],
    [('A1', 'A3'), None],
    [('A4', 'A1'), None],
    [('A2', 'A3'), None],

    [('B1', 'B2'), None],
    [('B3', 'B4'), None],
    [('B4', 'B2'), None],
    [('B1', 'B3'), None],
    [('B4', 'B1'), None],
    [('B2', 'B3'), None],

    [('C1', 'C2'), None],
    [('C3', 'C4'), None],
    [('C4', 'C2'), None],
    [('C1', 'C3'), None],
    [('C4', 'C1'), None],
    [('C2', 'C3'), None],

    [('D1', 'D2'), None],
    [('D3', 'D4'), None],
    [('D4', 'D2'), None],
    [('D1', 'D3'), None],
    [('D4', 'D1'), None],
    [('D2', 'D3'), None],

    [('E1', 'E2'), None],
    [('E3', 'E4'), None],
    [('E4', 'E2'), None],
    [('E1', 'E3'), None],
    [('E4', 'E1'), None],
    [('E2', 'E3'), None],

    [('F1', 'F2'), None],
    [('F3', 'F4'), None],
    [('F4', 'F2'), None],
    [('F1', 'F3'), None],
    [('F4', 'F1'), None],
    [('F2', 'F3'), None],

    [('G1', 'G2'), None],
    [('G3', 'G4'), None],
    [('G4', 'G2'), None],
    [('G1', 'G3'), None],
    [('G4', 'G1'), None],
    [('G2', 'G3'), None],

    [('H1', 'H2'), None],
    [('H3', 'H4'), None],
    [('H4', 'H2'), None],
    [('H1', 'H3'), None],
    [('H4', 'H1'), None],
    [('H2', 'H3'), None],

    [('I1', 'I2'), None],
    [('I3', 'I4'), None],
    [('I4', 'I2'), None],
    [('I1', 'I3'), None],
    [('I4', 'I1'), None],
    [('I2', 'I3'), None],

    [('J1', 'J2'), None],
    [('J3', 'J4'), None],
    [('J4', 'J2'), None],
    [('J1', 'J3'), None],
    [('J4', 'J1'), None],
    [('J2', 'J3'), None],

    [('K1', 'K2'), None],
    [('K3', 'K4'), None],
    [('K4', 'K2'), None],
    [('K1', 'K3'), None],
    [('K4', 'K1'), None],
    [('K2', 'K3'), None],

    [('L1', 'L2'), None],
    [('L3', 'L4'), None],
    [('L4', 'L2'), None],
    [('L1', 'L3'), None],
    [('L4', 'L1'), None],
    [('L2', 'L3'), None],
]

  # json.dump(group_stage_matches, f)
# with open("group_stage_matches.json") as f:
#   gsm = json.load(f)
# gsm

# %%
import json

def load_json(path):
  with open(path) as f:
    obj = json.load(f)
    return obj

def load_str(path):
  with open(path) as f:
    res = f.read()
    return res

# %%
group_stage_leaderboard = pd.read_csv("resources/data/group_stage_leaderboard.csv",  index_col=0)
group_stage_matches = load_json("resources/data/group_stage_matches.json")
knockout_stage_matches = load_str("resources/data/knockout_stage_matches.txt")
teams = group_stage_leaderboard.index
groups = load_json("resources/data/groups.json")
rankings = pd.read_csv("resources/data/fifa_rankings.csv", index_col=0)
outcomes_3rd = load_json("resources/data/3rd_place_outcomes.json")
team_codes = load_json("resources/data/team_codes.json")

# %%
def get_points(home_goals, away_goals):
  if home_goals == away_goals:
    return (1, 1)
  elif home_goals > away_goals:
    return (3, 0)
  else:
    return (0, 3)

def get_winner(home, home_goals, away, away_goals):
  if home_goals == away_goals:
    return 'draw'
  elif home_goals > away_goals:
    return home
  else:
    return away

import random
def simulate_match(home, away, knockout=False):
  """
  returns the goals scored in each match
  psuedo code:

  hg, ag = goal_probs(home, away)
  r = random()
  if hg == ag and knockout:
    return pens(home, away)
  return (hg, ag)
  """
  home_xg, away_xg = predict_xg(home, away)
  home_points = rankings.loc[home, 'points']
  away_points = rankings.loc[away, 'points']
  total_points = home_points + away_points
  p_home = home_points / total_points
  p_away = away_points / total_points
  r = random.random()
  if r < p_home:
  # if home_points > away_points:
    return (1, 0)
  else:
    return (0, 1)

# %%
rankings.loc["France", "points"]
simulate_match("France", "Argentina")

# %%
def sort_group(group):
  """FIFA WC'26 Tie-Breakers"""
  return group.sort_values(['points', 'gd', 'gf', 'rank'], ascending=False)

def placement(i):
  """Get place of exit in tournament by knockout match index"""
  if i < 16:
    return 32                 # Round of 32
  if i < 16 + 8:
    return 16                 # Round of 16
  if i < 16 + 8 + 4:
    return 8                  # Quarter Finals
  return 4                    # Semi Finals -> 1/2/3/4

def simulate_wc(*args):
  """
  Simulate the 2026 FIFA World Cup!
  """

  placements = {}

  lb = group_stage_leaderboard.copy()
  gs_matches = group_stage_matches.copy()

  # Group Stage
  for i, match in enumerate(gs_matches):
    t1, t2 = match[0]
    group = t1[0:1]
    home = groups[group][int(t1[1:2]) - 1]
    away = groups[group][int(t2[1:2]) - 1]
    hg, ag = simulate_match(home, away)
    hp, ap = get_points(hg, ag)
    lb.loc[home, 'points'] += hp
    lb.loc[home, 'gf'] += hg
    lb.loc[home, 'gd'] += hg - ag
    lb.loc[away, 'points'] += ap
    lb.loc[away, 'gf'] += ag
    lb.loc[away, 'gd'] += ag - hg
    gs_matches[i][1] = get_winner(home, hg, away, ag)

  # Knockout Rounds
  lb_groups = lb.groupby('group').apply(sort_group, include_groups=False)
  third_placers = lb_groups.iloc[2::4]
  qualified = third_placers.sort_values(['points', 'gd', 'gf', 'rank'],
                                        ascending=False)
  qualified = qualified.head(8).index.get_level_values(0)
  qualified = "".join(sorted([t[0] for t in qualified]))
  matchusps3rd1st = outcomes_3rd[qualified]
  _1A, _1B, _1D, _1E, _1G, _1I, _1K, _1L = 0, 1, 2, 3, 4, 5, 6, 7
  _3 = matchusps3rd1st
  ko_matches = eval(knockout_stage_matches)

  # Round of 32
  # for i in range(16):
  #   m = ko_matches[i]
    
  #   hg, ag = simulate_match(h, a, knockout=True)
  #   if hg > ag:
  #     ko_matches[i][1] = h
  #     ko_matches[i][2] = a
  #     placements[a] = "32"
  #   else:
  #     ko_matches[i][1] = a
  #     ko_matches[i][2] = h
  #     placements[h] = "32"

  # Round of 32 through Semi-Finals
  for i in range(30):
    m = ko_matches[i]
    if i < 16:
      h = m[0][0]
      h = lb_groups.loc[h[1:2]].index[int(h[0:1])-1]
      a = m[0][1]
      a = lb_groups.loc[a[1:2]].index[int(a[0:1])-1]
    else:
      h = ko_matches[m[0][0]][1]
      a = ko_matches[m[0][1]][1]
    hg, ag = simulate_match(h, a, knockout=True)
    if hg > ag:
      ko_matches[i][1] = h
      ko_matches[i][2] = a
      placements[a] = placement(i)
    else:
      ko_matches[i][1] = a
      ko_matches[i][2] = h
      placements[h] = placement(i)

  # 3rd-place match
  i = 30
  m = ko_matches[i]
  h = ko_matches[m[0][0]][2]
  a = ko_matches[m[0][1]][2]
  hg, ag = simulate_match(h, a, knockout=True)
  if hg > ag:
    ko_matches[i][1] = h
    ko_matches[i][2] = a
    # placements[h] = 3
    # placements[a] = 4
  else:
    ko_matches[i][1] = a
    ko_matches[i][2] = h
    # placements[a] = 3
    # placements[h] = 4

  # Final
  i = 31
  m = ko_matches[i]
  h = ko_matches[m[0][0]][1]
  a = ko_matches[m[0][1]][1]
  hg, ag = simulate_match(h, a, knockout=True)
  if hg > ag:
    ko_matches[i][1] = h
    ko_matches[i][2] = a
    placements[h] = 1
    placements[a] = 2
  else:
    ko_matches[i][1] = a
    ko_matches[i][2] = h
    placements[a] = 1
    placements[h] = 2

  # global teams
  lb = lb.drop(columns=['group'])
  for team in teams:
    if team not in placements:
      placements[team] = 48
    p = placements[team]
    placements[team] = (lb.loc[team], p)

  return placements

from timeit import default_timer as timer
start = timer()
res = simulate_wc()
end = timer()
print(end - start)
display(res)

# %%
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

def run_simulations(num_simulations, max_workers=None):
  """
  Run `num_simulations` World Cup Simulations
  """
  n = num_simulations

  with ProcessPoolExecutor(max_workers=max_workers) as executor:
    all_placements = list(executor.map(simulate_wc, range(n)))

  counts = defaultdict(lambda: defaultdict(int))
  gs_lb = group_stage_leaderboard.copy().drop(columns=['group'])

  for result in all_placements:
    for team, [gs_results, place] in result.items():
      gs_lb.loc[team] += gs_results
      # if place < 48:  counts[team]["Group Stage"] += 1
      if place <= 32: counts[team]["Round of 32"] += 1
      if place <= 16: counts[team]["Round of 16"] += 1
      if place <= 8:  counts[team]["Quarter-Finals"] += 1
      if place <= 4:  counts[team]["Semi-Finals"] += 1
      if place <= 2:  counts[team]["Finals"] += 1
      if place == 1:  counts[team]["Win"] += 1

  df = pd.DataFrame.from_dict(counts, orient="index").fillna(0).astype(int)
  df = df[["Win", "Finals", "Semi-Finals", "Quarter-Finals",
           "Round of 16", "Round of 32"]] #, "Group Stage"]]
  df  = df * 100.0 / n
  df = df.sort_values("Win", ascending=False)

  gs_lb = gs_lb / n

  return df, gs_lb

from timeit import default_timer as timer
start = timer()
placements, avg_gs_lb = run_simulations(10**3)
end = timer()
print(end - start)
display(placements)
display(avg_gs_lb)

# %%
# 10**3 -> ~8s
# 10**4 -> ~80s
# 10**5 -> ~830s
# display(placements)
# display(avg_gs_lb)
