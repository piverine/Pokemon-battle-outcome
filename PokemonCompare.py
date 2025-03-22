import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests

def get_pokemon_image(pokemon_name):
    """Fetch Pokémon image from the PokeAPI."""
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return {
            "default": data["sprites"]["front_default"],  # Regular front sprite
            "shiny": data["sprites"]["front_shiny"],      # Shiny version
            "official": data["sprites"]["other"]["official-artwork"]["front_default"]  # High-res official artwork
        }
    else:
        return None

def get_pokemon_data(pokemon_name):
    """Fetch Pokémon data from the dataset by name."""
    pokemon_data = df[df["Name"].str.lower() == pokemon_name.lower()]
    return pokemon_data if not pokemon_data.empty else None

def get_effectiveness(attacking_type, defending_type):
    """Returns the effectiveness multiplier for an attacking type against a defending type."""
    type_chart = {
        "Normal":    {"Rock": 0.5, "Ghost": 0, "Steel": 0.5},
        "Fire":      {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": 0.5, "Dragon": 0.5, "Steel": 2},
        "Water":     {"Fire": 2, "Water": 0.5, "Grass": 0.5, "Ground": 2, "Rock": 2, "Dragon": 0.5},
        "Electric":  {"Water": 2, "Electric": 0.5, "Grass": 0.5, "Ground": 0, "Flying": 2, "Dragon": 0.5},
        "Grass":     {"Fire": 0.5, "Water": 2, "Grass": 0.5, "Poison": 0.5, "Ground": 2, "Flying": 0.5, "Bug": 0.5, "Rock": 2, "Dragon": 0.5, "Steel": 0.5},
        "Ice":       {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 0.5, "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": 0.5},
        "Fighting":  {"Normal": 2, "Ice": 2, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Rock": 2, "Ghost": 0, "Dark": 2, "Steel": 2, "Fairy": 0.5},
        "Poison":    {"Grass": 2, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5, "Steel": 0, "Fairy": 2},
        "Ground":    {"Fire": 2, "Electric": 2, "Grass": 0.5, "Poison": 2, "Flying": 0, "Bug": 0.5, "Rock": 2, "Steel": 2},
        "Flying":    {"Electric": 0.5, "Grass": 2, "Fighting": 2, "Bug": 2, "Rock": 0.5, "Steel": 0.5},
        "Psychic":   {"Fighting": 2, "Poison": 2, "Psychic": 0.5, "Dark": 0, "Steel": 0.5},
        "Bug":       {"Fire": 0.5, "Grass": 2, "Fighting": 0.5, "Poison": 0.5, "Flying": 0.5, "Psychic": 2, "Ghost": 0.5, "Dark": 2, "Steel": 0.5, "Fairy": 0.5},
        "Rock":      {"Fire": 2, "Ice": 2, "Fighting": 0.5, "Ground": 0.5, "Flying": 2, "Bug": 2, "Steel": 0.5},
        "Ghost":     {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": 0.5},
        "Dragon":    {"Dragon": 2, "Steel": 0.5, "Fairy": 0},
        "Dark":      {"Fighting": 0.5, "Psychic": 2, "Ghost": 2, "Dark": 0.5, "Fairy": 0.5},
        "Steel":     {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2, "Rock": 2, "Steel": 0.5, "Fairy": 2},
        "Fairy":     {"Fighting": 2, "Poison": 0.5, "Steel": 0.5, "Dragon": 2, "Dark": 2}
    }
    
    return type_chart.get(attacking_type, {}).get(defending_type, 1)  # Default is 1 (neutral)

def calculate_tas(pokemon_a, pokemon_b):
    type1_a, type2_a = pokemon_a
    type1_b, type2_b = pokemon_b

    if type2_a is None:
        type2_a = type1_a
    if type2_b is None:
        type2_b = type1_b

    effectiveness_a = (
        get_effectiveness(type1_a, type1_b) + 
        get_effectiveness(type1_a, type2_b) + 
        get_effectiveness(type2_a, type1_b) + 
        get_effectiveness(type2_a, type2_b)
    ) / 4

    # Calculate effectiveness of B attacking A
    effectiveness_b = (
        get_effectiveness(type1_b, type1_a) + 
        get_effectiveness(type1_b, type2_a) + 
        get_effectiveness(type2_b, type1_a) + 
        get_effectiveness(type2_b, type2_a)
    ) / 4

    # Type Advantage Score
    tas = effectiveness_a - effectiveness_b
    return tas

def calculate_win_probability(pokemon1_data, pokemon2_data):
    """
    Calculate the win probability for Pokemon 1 against Pokemon 2.
    Returns a percentage value (0-100).
    """
    # Get types for type advantage calculation
    pokemon_a = (pokemon1_data['Type 1'].iloc[0], pokemon1_data['Type 2'].iloc[0])
    pokemon_b = (pokemon2_data['Type 1'].iloc[0], pokemon2_data['Type 2'].iloc[0])
    
    # Calculate type advantage score
    tas = calculate_tas(pokemon_a, pokemon_b)
    
    # Extract relevant stats for both Pokémon
    stats1 = {
        'hp': pokemon1_data['HP'].iloc[0],
        'attack': pokemon1_data['Attack'].iloc[0],
        'defense': pokemon1_data['Defense'].iloc[0],
        'sp_attack': pokemon1_data['Sp. Atk'].iloc[0],
        'sp_defense': pokemon1_data['Sp. Def'].iloc[0],
        'speed': pokemon1_data['Speed'].iloc[0],
        'total': pokemon1_data['Total'].iloc[0]
    }
    
    stats2 = {
        'hp': pokemon2_data['HP'].iloc[0],
        'attack': pokemon2_data['Attack'].iloc[0],
        'defense': pokemon2_data['Defense'].iloc[0],
        'sp_attack': pokemon2_data['Sp. Atk'].iloc[0],
        'sp_defense': pokemon2_data['Sp. Def'].iloc[0],
        'speed': pokemon2_data['Speed'].iloc[0],
        'total': pokemon2_data['Total'].iloc[0]
    }
    
    # Calculate stat ratios (Pokemon1 / Pokemon2)
    stat_ratios = {
        'hp_ratio': stats1['hp'] / max(1, stats2['hp']),
        'attack_ratio': stats1['attack'] / max(1, stats2['defense']),  # Attack vs opposing Defense
        'defense_ratio': stats1['defense'] / max(1, stats2['attack']),  # Defense vs opposing Attack
        'sp_attack_ratio': stats1['sp_attack'] / max(1, stats2['sp_defense']),  # Sp. Attack vs opposing Sp. Defense
        'sp_defense_ratio': stats1['sp_defense'] / max(1, stats2['sp_attack']),  # Sp. Defense vs opposing Sp. Attack
        'speed_ratio': stats1['speed'] / max(1, stats2['speed']),
        'total_ratio': stats1['total'] / max(1, stats2['total'])
    }
    
    # Assign weights to different components
    weights = {
        'type_advantage': 0.30,  # 30% based on type matchup
        'hp_ratio': 0.10,
        'attack_ratio': 0.10,
        'defense_ratio': 0.10,
        'sp_attack_ratio': 0.10,
        'sp_defense_ratio': 0.10,
        'speed_ratio': 0.10,
        'total_ratio': 0.10
    }
    
    # Convert tas to a 0-1 scale (from potentially -2 to 2)
    normalized_tas = (tas + 2) / 4
    
    # Calculate weighted score
    weighted_score = (
        weights['type_advantage'] * normalized_tas +
        weights['hp_ratio'] * min(2, stat_ratios['hp_ratio']) / 2 +
        weights['attack_ratio'] * min(2, stat_ratios['attack_ratio']) / 2 +
        weights['defense_ratio'] * min(2, stat_ratios['defense_ratio']) / 2 +
        weights['sp_attack_ratio'] * min(2, stat_ratios['sp_attack_ratio']) / 2 +
        weights['sp_defense_ratio'] * min(2, stat_ratios['sp_defense_ratio']) / 2 +
        weights['speed_ratio'] * min(2, stat_ratios['speed_ratio']) / 2 +
        weights['total_ratio'] * min(2, stat_ratios['total_ratio']) / 2
    )
    
    # Convert to win probability percentage (between 0 and 100)
    win_probability = min(100, max(0, weighted_score * 100))
    
    return round(win_probability, 1)

# Streamlit Setup
st.set_page_config(
    page_title="Pokemon",
    page_icon="✨",
    layout="centered"
)

# Add a title
st.title("Pokemon")

# Add some text
st.write("Pokemon win-probability predictor")

# Load data
df = pd.read_csv("Pokemon.csv")
st.dataframe(df)

# User input for Pokemon names
pokemon1_name = st.text_input("Pokemon 1:", key="pokemon1")
pokemon2_name = st.text_input("Pokemon 2:", key="pokemon2")

# Layout for displaying Pokémon and their images
col1, vs_col, col2 = st.columns([4, 1, 4])

with col1:
    st.write("Pokemon 1: " + pokemon1_name)
    if pokemon1_name:
        images = get_pokemon_image(pokemon1_name)
        if images:
            st.image(images["official"], caption=f"{pokemon1_name.capitalize()}")
        else:
            st.error("Pokémon not found. Check the name and try again!")

with vs_col:
    st.markdown("<h1 style='text-align: center; color: red; margin-top: 200px;'>VS</h1>", unsafe_allow_html=True)

with col2:
    st.write("Pokemon 2: " + pokemon2_name)
    if pokemon2_name:
        images = get_pokemon_image(pokemon2_name)
        if images:
            st.image(images["official"], caption=f"{pokemon2_name.capitalize()}")
        else:
            st.error("Pokémon not found. Check the name and try again!")

# Compare button logic
if st.button("Compare"):
    # Fetch Pokémon data only when button is clicked
    pokemon1_data = get_pokemon_data(pokemon1_name)
    pokemon2_data = get_pokemon_data(pokemon2_name)

    if pokemon1_data is not None and pokemon2_data is not None:
        pokemon_a = (pokemon1_data['Type 1'].iloc[0], pokemon1_data['Type 2'].iloc[0])
        pokemon_b = (pokemon2_data['Type 1'].iloc[0], pokemon2_data['Type 2'].iloc[0])

        # Calculate type advantage score
        tas = calculate_tas(pokemon_a, pokemon_b)
        st.write(f"Type Advantage Score: {tas}")
        
        # Calculate win probability
        win_prob = calculate_win_probability(pokemon1_data, pokemon2_data)
        
        # Display the win probability
        st.write(f"**Win Probability for {pokemon1_name}**: {win_prob}%")
        st.progress(win_prob/100)
        
        # Extract stats for visualization
        stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        
        # Create comparison dataframe for bar chart
        chart_data = pd.DataFrame({
            'Stat': stats + stats,  # Repeat stats for both Pokémon
            'Value': [
                pokemon1_data['HP'].iloc[0], pokemon1_data['Attack'].iloc[0], 
                pokemon1_data['Defense'].iloc[0], pokemon1_data['Sp. Atk'].iloc[0], 
                pokemon1_data['Sp. Def'].iloc[0], pokemon1_data['Speed'].iloc[0],
                pokemon2_data['HP'].iloc[0], pokemon2_data['Attack'].iloc[0], 
                pokemon2_data['Defense'].iloc[0], pokemon2_data['Sp. Atk'].iloc[0], 
                pokemon2_data['Sp. Def'].iloc[0], pokemon2_data['Speed'].iloc[0]
            ],
            'Pokémon': [pokemon1_name] * 6 + [pokemon2_name] * 6  # Identify which Pokémon each stat belongs to
        })
        
        # Create a grouped bar chart
        st.subheader("Stats Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use Seaborn for a grouped bar chart with custom styling
        sns.barplot(x='Stat', y='Value', hue='Pokémon', data=chart_data, palette=['#ff9999', '#66b3ff'])
        
        # Customize the chart
        plt.title(f"{pokemon1_name} vs {pokemon2_name} Stats", fontsize=16)
        plt.ylabel("Stat Value", fontsize=12)
        plt.xlabel("", fontsize=12)  # Hide x-axis label since it's obvious
        plt.xticks(rotation=0)  # Keep stat names horizontal
        plt.legend(title="")
        
        # Add value labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'bottom', 
                      fontsize=10)
        
        # Display the bar chart
        st.pyplot(fig)
        
        # Create a pie chart for win probability
        st.subheader("Win Probability")
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        
        # Data for the pie chart
        win_data = [win_prob, 100 - win_prob]
        labels = [f"{pokemon1_name} ({win_prob}%)", f"{pokemon2_name} ({100-win_prob}%)"]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.1, 0)  # Explode the first slice for emphasis
        
        # Create the pie chart
        ax2.pie(win_data, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', 
               shadow=True, startangle=90, textprops={'fontsize': 14})
        ax2.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        
        # Add a title
        plt.title("Battle Outcome Probability", fontsize=16)
        
        # Display the pie chart
        st.pyplot(fig2)
        
        # Display the total stats comparison
        st.subheader("Total Stats")
        total1 = pokemon1_data['Total'].iloc[0]
        total2 = pokemon2_data['Total'].iloc[0]
        
        # Create a comparison bar for total stats
        total_data = pd.DataFrame({
            'Pokémon': [pokemon1_name, pokemon2_name],
            'Total': [total1, total2]
        })
        
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.barplot(x='Pokémon', y='Total', data=total_data, palette=['#ff9999', '#66b3ff'])
        
        # Add value labels on top of each bar
        for p in ax3.patches:
            ax3.annotate(f'{int(p.get_height())}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'bottom', 
                      fontsize=12)
        
        plt.title("Total Stats Comparison", fontsize=16)
        plt.ylabel("Total Stat Value", fontsize=12)
        
        # Display the total stats chart
        st.pyplot(fig3)
        
    else:
        st.error("Please enter valid Pokemon names")
