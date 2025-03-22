import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

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

    # Handle None types (single-type Pokémon)
    if type2_a is None:
        type2_a = type1_a  # Treat it as duplicate of Type1 for averaging
    if type2_b is None:
        type2_b = type1_b

    # Calculate effectiveness of A attacking B
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

def radar_chart(fig, categories, values, title, position, color, label):
    """Create a radar chart for Pokémon stats."""
    # Number of categories
    N = len(categories)
    
    # Create angles for each category (evenly spaced)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Make the plot circular by repeating the first value and angle
    values += values[:1]
    angles += angles[:1]
    
    # Create subplot
    ax = fig.add_subplot(position, polar=True)
    
    # Draw one axis per category and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Set y limits
    ax.set_ylim(0, 250)
    
    # Plot data
    ax.plot(angles, values, color=color, linewidth=2, label=label)
    
    # Fill area
    ax.fill(angles, values, color=color, alpha=0.25)
    
    # Add title
    ax.set_title(title, size=14, color=color, y=1.1)
    
    return ax

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
        
        # Create a tab system for different visualizations
        tab1, tab2, tab3 = st.tabs(["Radar Chart", "Stat Distribution", "Win Probability"])
        
        with tab1:
            # Create a radar chart comparing both Pokémon
            st.subheader("Stats Comparison - Radar Chart")
            
            # Prepare data for radar chart
            stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            
            # Get stats values for both Pokémon
            stats1 = [
                pokemon1_data['HP'].iloc[0], 
                pokemon1_data['Attack'].iloc[0], 
                pokemon1_data['Defense'].iloc[0], 
                pokemon1_data['Sp. Atk'].iloc[0], 
                pokemon1_data['Sp. Def'].iloc[0], 
                pokemon1_data['Speed'].iloc[0]
            ]
            
            stats2 = [
                pokemon2_data['HP'].iloc[0], 
                pokemon2_data['Attack'].iloc[0], 
                pokemon2_data['Defense'].iloc[0], 
                pokemon2_data['Sp. Atk'].iloc[0], 
                pokemon2_data['Sp. Def'].iloc[0], 
                pokemon2_data['Speed'].iloc[0]
            ]
            
            # Create figure with two radar charts
            fig = plt.figure(figsize=(12, 6))
            
            # Draw radar chart for Pokémon 1
            radar_chart(fig, stats, stats1, pokemon1_name, 121, '#ff5959', pokemon1_name)
            
            # Draw radar chart for Pokémon 2
            radar_chart(fig, stats, stats2, pokemon2_name, 122, '#5959ff', pokemon2_name)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add a combined radar chart for direct comparison
            st.subheader("Head-to-Head Comparison")
            fig2 = plt.figure(figsize=(10, 8))
            ax = radar_chart(fig2, stats, stats1, f"{pokemon1_name} vs {pokemon2_name}", 111, '#ff5959', pokemon1_name)
            radar_chart(fig2, stats, stats2, "", 111, '#5959ff', pokemon2_name)
            
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        with tab2:
            # Create a lollipop chart for comparing total stats
            st.subheader("Stat Distribution")
            
            # Prepare data for visualization
            stat_names = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            
            # Create DataFrame with all stats
            stats_df = pd.DataFrame({
                'Stat': stat_names * 2,
                'Value': stats1 + stats2,
                'Pokémon': [pokemon1_name] * 6 + [pokemon2_name] * 6
            })
            
            # Create swarm plot with points and connecting lines
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Draw the scatter points
            sns.swarmplot(x='Stat', y='Value', hue='Pokémon', data=stats_df, 
                         palette=['#ff5959', '#5959ff'], size=10, ax=ax)
            
            # Add horizontal lines connecting points for each Pokémon
            for pokemon in [pokemon1_name, pokemon2_name]:
                pokemon_stats = stats_df[stats_df['Pokémon'] == pokemon]
                stat_values = pokemon_stats['Value'].values
                x_positions = np.arange(len(stat_names))
                
                color = '#ff5959' if pokemon == pokemon1_name else '#5959ff'
                ax.plot(x_positions, stat_values, 'o-', color=color, alpha=0.7, linewidth=2)
            
            # Customize plot
            plt.title("Stat Comparison", fontsize=16)
            plt.ylabel("Value", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Add a violin plot showing stat distributions
            st.subheader("Stat Distribution Patterns")
            
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            sns.violinplot(x='Stat', y='Value', hue='Pokémon', data=stats_df, 
                          palette=['#ff5959', '#5959ff'], split=True, inner='quart',
                          linewidth=1, ax=ax3)
            
            # Add stripplot for individual points
            sns.stripplot(x='Stat', y='Value', hue='Pokémon', data=stats_df, 
                         palette=['#800000', '#000080'], size=5, jitter=True, 
                         dodge=True, ax=ax3, alpha=0.7)
            
            # Remove legend from stripplot (keep only one legend)
            handles, labels = ax3.get_legend_handles_labels()
            ax3.legend(handles[:2], labels[:2], title="Pokémon")
            
            plt.title("Stat Distributions", fontsize=16)
            plt.tight_layout()
            st.pyplot(fig3)
        
        with tab3:
            # Create pie chart for win probability
            st.subheader("Win Probability")
            
            # Set up columns for displaying multiple charts
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Standard pie chart
                fig4, ax4 = plt.subplots(figsize=(8, 8))
                
                # Data for the pie chart
                win_data = [win_prob, 100 - win_prob]
                labels = [f"{pokemon1_name}", f"{pokemon2_name}"]
                colors = ['#ff5959', '#5959ff']
                explode = (0.1, 0)  # Explode the first slice for emphasis
                
                # Create the pie chart
                wedges, texts, autotexts = ax4.pie(
                    win_data, 
                    explode=explode, 
                    labels=labels, 
                    colors=colors, 
                    autopct='%1.1f%%', 
                    shadow=True, 
                    startangle=90, 
                    textprops={'fontsize': 14}
                )
                
                # Style the percentage text
                for autotext in autotexts:
                    autotext.set_fontweight('bold')
                
                ax4.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
                plt.title("Battle Outcome Probability", fontsize=16)
                st.pyplot(fig4)
            
            with col2:
                # Create gauge chart for win probability
                fig5 = plt.figure(figsize=(8, 8))
                ax5 = fig5.add_subplot(111, projection='polar')
                
                # Gauge chart parameters
                theta = np.linspace(0, 2 * np.pi, 100)
                radii = np.full_like(theta, 0.9)
                
                # Background circle
                ax5.plot(theta, radii, color='#d9d9d9', linewidth=30, alpha=0.5)
                
                # Probability arc
                probability_angle = 2 * np.pi * (win_prob / 100)
                probability_theta = np.linspace(0, probability_angle, 100)
                probability_radii = np.full_like(probability_theta, 0.9)
                ax5.plot(probability_theta, probability_radii, color='#ff5959', linewidth=30)
                
                # Rest of the circle
                rest_theta = np.linspace(probability_angle, 2 * np.pi, 100)
                rest_radii = np.full_like(rest_theta, 0.9)
                ax5.plot(rest_theta, rest_radii, color='#5959ff', linewidth=30, alpha=1.0 if win_prob < 100 else 0)
                
                # Style the chart
                ax5.set_rticks([])  # Remove radial ticks
                ax5.set_xticks([])  # Remove angular ticks
                
                # Add text in the middle
                ax5.text(0, 0, f"{win_prob}%", fontsize=36, ha='center', va='center', fontweight='bold', color='#333333')
                ax5.text(0, -0.3, f"{pokemon1_name}", fontsize=16, ha='center', va='center', color='#ff5959')
                ax5.text(0, 0.3, f"{100-win_prob}%\n{pokemon2_name}", fontsize=16, ha='center', va='center', color='#5959ff')
                
                plt.title("Win Probability Gauge", fontsize=16, pad=20)
                st.pyplot(fig5)
        
        # Display a heatmap of stats
        st.subheader("Stats Heatmap")
        
        # Create a DataFrame for the heatmap
        heatmap_data = pd.DataFrame({
            'Stat': ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],
            pokemon1_name: [
                pokemon1_data['HP'].iloc[0], 
                pokemon1_data['Attack'].iloc[0], 
                pokemon1_data['Defense'].iloc[0], 
                pokemon1_data['Sp. Atk'].iloc[0], 
                pokemon1_data['Sp. Def'].iloc[0], 
                pokemon1_data['Speed'].iloc[0]
            ],
            pokemon2_name: [
                pokemon2_data['HP'].iloc[0], 
                pokemon2_data['Attack'].iloc[0], 
                pokemon2_data['Defense'].iloc[0], 
                pokemon2_data['Sp. Atk'].iloc[0], 
                pokemon2_data['Sp. Def'].iloc[0], 
                pokemon2_data['Speed'].iloc[0]
            ]
        })
        
        # Set the 'Stat' column as index
        heatmap_data = heatmap_data.set_index('Stat')
        
        # Create heatmap
        fig6, ax6 = plt.subplots(figsize=(10, 5))
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt='d', linewidths=.5, ax=ax6)
        plt.title("Stats Comparison Heatmap", fontsize=16)
        plt.tight_layout()
        st.pyplot(fig6)
        
    else:
        st.error("Please enter valid Pokemon names")