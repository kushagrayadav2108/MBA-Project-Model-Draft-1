import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seed_demand_model import train_and_evaluate
import os

# Set style for academic paper
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def generate_visualizations():
    print("Training model to extract coefficients...")
    try:
        model = train_and_evaluate('Mba BA project.xlsx')
    except Exception as e:
        print(f"Error training model: {e}")
        return

    # Extract coefficients
    coef_data = []
    varieties = []
    
    # Feature names in order
    features = [
        'Max Temp', 'Min Temp', 'Pre-Monsoon Rain', 
        'Monsoon Rain', 'Post-Monsoon Rain', 'Monsoon Duration'
    ]
    
    print("\nExtracting coefficients for varieties with trained models:")
    for variety, reg_model in model.models.items():
        print(f"- {variety}")
        varieties.append(variety.replace('_', '-'))
        coef_data.append(reg_model.coef_)
    
    if not coef_data:
        print("No models were trained successfully (insufficient data). Cannot generate coefficient plot.")
        return

    # Create DataFrame for plotting
    coef_df = pd.DataFrame(coef_data, columns=features, index=varieties)
    
    # 1. Coefficient Heatmap
    plt.figure(figsize=(12, 6))
    
    # Create distinct colormap (diverging)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Draw heatmap
    ax = sns.heatmap(coef_df, annot=True, fmt=".3f", cmap=cmap, center=0,
                     linewidths=.5, cbar_kws={"label": "Standardized Regression Coefficient"})
    
    plt.title('Impact of Meteorological Factors on Rice Variety Share\n(Ridge Regression Coefficients)', pad=20, fontsize=14, weight='bold')
    plt.ylabel('Variety', fontsize=12)
    plt.xlabel('Meteorological Factor', fontsize=12)
    
    # Rotate x-axis labels
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    output_path = 'model_coefficients_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved coefficient heatmap to: {os.path.abspath(output_path)}")
    
    # 2. Bar Chart for Key Varieties (Pb-1121 and Pb-1509)
    if 'Pb-1121' in varieties and 'Pb-1509' in varieties:
        plt.figure(figsize=(14, 7))
        
        # Melt dataframe for bar plot
        plot_df = coef_df.loc[['Pb-1121', 'Pb-1509']].reset_index().melt(id_vars='index', var_name='Factor', value_name='Coefficient')
        plot_df.rename(columns={'index': 'Variety'}, inplace=True)
        
        sns.barplot(data=plot_df, x='Factor', y='Coefficient', hue='Variety', palette='viridis')
        
        plt.axhline(0, color='black', linewidth=1)
        plt.title('Comparison of Meteorological Sensitivity:\nLong Duration (Pb-1121) vs Short Duration (Pb-1509)', pad=20, fontsize=14, weight='bold')
        plt.xticks(rotation=20, ha='right')
        plt.ylabel('Standardized Coefficient (Impact Strength)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Variety')
        
        bar_output_path = 'variety_sensitivity_comparison.png'
        plt.savefig(bar_output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison bar chart to: {os.path.abspath(bar_output_path)}")

if __name__ == "__main__":
    generate_visualizations()
