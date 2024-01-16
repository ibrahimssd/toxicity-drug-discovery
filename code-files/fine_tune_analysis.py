import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import logging
import os
import ast
import datetime
import pandas as pd
from scipy.stats import mannwhitneyu
# import seaborn as sns
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# plot 1: Compare atom-wise tokenizers (SELFIES and SMILES) based on score
def plot_atom_wise_performance(performance_df, score, file_name):
    """
    Plots the performance of atom-wise tokenizers based on the given score.
    
    :param performance_df: DataFrame containing the performance data.
    :param score: The score type to be plotted.
    :param file_name: The suffix of the file name where the plot will be saved.
    """
    atom_wise_df = performance_df[performance_df['tokenizer_type'] == 'Atom-wise']
    sorted_atom_wise_df = atom_wise_df.sort_values(by='mol_rep')
    max_score = atom_wise_df.groupby('mol_rep').max()[score]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(max_score)))
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    bars = ax.bar(max_score.index, max_score, color=colors)
    
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(
            f'score: {height:.3f}', 
            xy=(bar.get_x() + bar.get_width() / 2, height), 
            xytext=(-10, 5),  # Offset text for better readability
            textcoords="offset points", 
            ha='right', 
            va='bottom',
            fontweight='bold',
            fontsize=10
        )
        ax.annotate(
            f'vocab: {sorted_atom_wise_df["vocab_size"].iloc[idx]}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(-10, 20),  # Offset text for better readability
            textcoords="offset points",
            ha='right',
            va='bottom',
            fontweight='bold',
            fontsize=10

        )

    # plot error bars 
    yerr_lower = atom_wise_df[score] - atom_wise_df[f'{score}_lb']
    yerr_upper = atom_wise_df[f'{score}_ub'] - atom_wise_df[score]
    ax.errorbar(atom_wise_df['mol_rep'], atom_wise_df[score], yerr=[yerr_lower.values, yerr_upper.values], fmt='o', color='black', capsize=5)

    # Configuring plot appearance
    plt.ylabel('AUC ROC', fontsize=12, fontweight='bold', labelpad=15)
    plt.xlabel('Molecule Representation', fontsize=12, labelpad=15, fontweight='bold')
    plt.title(f'Performance of Atom-wise Tokenizers on Task {atom_wise_df["task"].iloc[0]}', fontsize=14)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.xticks(rotation=45, fontsize=11, ha='right', fontweight='bold')
    
    # Save the enhanced plot
    plt.tight_layout()  # Ensure everything fits without overlap
    plt.savefig(f'./plots/scores_performance/{score}_atom_wise{file_name}.png')
    
    # close the plot
    plt.close()


# Plot 2: Compare tokenizers based on score at best vocab
def plot_best_vocabs_performance(performance_df, score, file_name):
    """
    Plots the performance of tokenizers based on the given score at their best vocab size.
    
    Parameters:
    - performance_df: DataFrame containing the performance data.
    - score: The score type to be plotted.
    - file_name: The suffix of the file name where the plot will be saved.
    """
    
    # Extract relevant data
    tokenizer_best_vocabs = performance_df.loc[
        performance_df.groupby(['tokenizer_type','mol_rep'])[score].idxmax()
    ][['tokenizer_type', 'vocab_size', 'mol_rep', score, f'{score}_lb', f'{score}_ub']]
    
    tokenizer_best_vocabs = tokenizer_best_vocabs.reset_index(drop=True)
    sorted_data = tokenizer_best_vocabs.sort_values(by=score, ascending=False)
    
    # Define colors
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(sorted_data)))
    
    # Create the plot
    plt.figure(figsize=(14, 12))
    ax = plt.gca()
    
    # Define x-values and error bars
    x_values = list(zip(sorted_data['tokenizer_type'], sorted_data['mol_rep']))
    yerr_lower = sorted_data[score] - sorted_data[f'{score}_lb']
    yerr_upper = sorted_data[f'{score}_ub'] - sorted_data[score]
    
    # Plot bars with error bars
    bars = ax.bar(
        range(len(x_values)), 
        sorted_data[score], 
        color=colors, 
        # yerr=[yerr_lower.values, yerr_upper.values], 
        capsize=5,
        width=0.6  # Reduced bar width
    )
    
    # Annotate bars with score and vocab size
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(
            f'score:  {height:.3f}', 
            xy=(bar.get_x() + bar.get_width() / 2, height), 
            xytext=(0, 5),  # Moved text up
            textcoords="offset points", 
            ha='center', 
            va='bottom',
            fontweight='bold',
            fontsize=10
        )
        ax.annotate(
            f'vocab:  {sorted_data["vocab_size"].iloc[idx]}', 
            xy=(bar.get_x() + bar.get_width() / 2, height), 
            xytext=(0, 20),  # Moved text up
            textcoords="offset points", 
            ha='center', 
            va='bottom',
            fontweight='bold',
            fontsize=10
        )
    
    # Set labels, title, and grid
    plt.ylabel(f'AUC ROC', fontsize=24)
    plt.xlabel('Tokenizer Type and Corresponding Molecular Representation', fontsize=24)
    plt.title(f'Performance of Tokenizers at their Best Vocab Size on Task {performance_df["task"].iloc[0]}', fontsize=24)
    plt.xticks(range(len(x_values)), x_values, rotation=10, fontweight='bold', fontsize=12)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6) 
    
    # Save and display the plot
    plt.savefig(f'./plots/scores_performance/{score}_tokenizer_best_vocabs{file_name}.png')
    plt.close()



# plot 3.1: Compare vocab_sizes performance for  subword tokenizers
def plot_vocabulary_performance(performance_df, score, file_name):
    """
    Plot the performance of subword tokenizers based on vocabulary size.
    
    Parameters:
    - performance_df: DataFrame containing tokenizer performance data.
    - score: The score column to plot.
    - file_name: Name of the file to save the plot.
    - save: Whether to save the plot or display it.
    """
    
    def annotate_scores(ax, df, score):
        for x, y in zip(df['vocab_size'].values, df[score].values):
            ax.text(x, y + 0.005, f"{y:.3f}", ha="center", va="bottom", fontsize=10, color='black')
            # ax.text(x, ax.get_ylim()[0], f"{x}", ha="center", va="top", fontsize=10, color='black')

    def plot_error_bars(ax, df, score, color):
        yerr_lower = df[score] - df[f'{score}_lb']
        yerr_upper = df[f'{score}_ub'] - df[score]
        ax.errorbar(df['vocab_size'], df[score], yerr=[yerr_lower.values, yerr_upper.values], fmt='-o', color=color)

    def plot_atomic_scores(ax, atomic_score):
        atomic_colors = ['red', 'blue']
        for idx, (mol_rep, atomic_s) in enumerate(atomic_score.iteritems()):
            ax.axhline(y=atomic_s, color=atomic_colors[idx], linestyle='--', alpha=0.5, label=f'{mol_rep} Atomic Score')
            # ax.text(ax.get_xlim()[0], atomic_s, f"{atomic_s:.3f}", va="center", ha="right", color=atomic_colors[idx], fontsize=10, fontweight='bold')

    def configure_plot(ax, df, score):
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.xscale('log')
        plt.xticks(df['vocab_size'], labels=[str(v) for v in df['vocab_size']], rotation=45)
        plt.xlabel('Vocab Size (log scale)', fontsize=14, fontweight='bold')
        plt.ylabel(score, fontsize=14, fontweight='bold')
        plt.title(f'AUC ROC Performance of Subword Tokenizers', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)

    # Main plotting logic
    sub_tokenizers = performance_df[performance_df['tokenizer_type'] != 'Atom-wise']['tokenizer_type'].unique()
    atomic_score = performance_df[performance_df['tokenizer_type'] == 'Atom-wise'].groupby('mol_rep').max()[score]
    
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    colors = plt.cm.viridis(np.linspace(0, 1, len(sub_tokenizers)))
    
    for idx, tokenizer in enumerate(sub_tokenizers):
        tokenizer_df = performance_df[performance_df['tokenizer_type'] == tokenizer].sort_values(by='vocab_size')
        plt.plot(tokenizer_df['vocab_size'], tokenizer_df[score], '-o', label=tokenizer, color=colors[idx])
        annotate_scores(ax, tokenizer_df, score)
        plot_error_bars(ax, tokenizer_df, score, colors[idx])
    
    plot_atomic_scores(ax, atomic_score)
    configure_plot(ax, performance_df, score)

    plt.tight_layout()
    plt.savefig(f'./plots/scores_performance/{score}_subword_tokenizers{file_name}.png')
    # close the plot
    plt.close()



# plot 3.2: Compare vocab_sizes performance for  subword tokenizers
def plot_vocabulary_performance_pinpoints(performance_df, score, file_name):
    """
    Plot the performance of subword tokenizers based on vocabulary size using a pinpoint plot.
    Group the points within each vocab size.
    
    Parameters:
    - performance_df: DataFrame containing tokenizer performance data.
    - score: The score column to plot.
    - file_name: Name of the file to save the plot.
    """
    
    def annotate_scores(ax, x, y_values, labels):
        for y, label in zip(y_values, labels):
            ax.text(x, y + 0.005, f"{label:.3f}", ha="center", va="bottom", fontsize=10, color='black')

    def plot_atomic_scores(ax, atomic_score):
        atomic_colors = ['red', 'blue']
        for idx, (mol_rep, atomic_s) in enumerate(atomic_score.iteritems()):
            ax.axhline(y=atomic_s, color=atomic_colors[idx], linestyle='--', alpha=0.5, label=f'{mol_rep} Atomic Score')
            # ax.text(ax.get_xlim()[0], atomic_s, f"{atomic_s:.3f}", va="center", ha="right", color=atomic_colors[idx], fontsize=10, fontweight='bold')

    def configure_plot(ax, df, score):
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.xscale('log')
        plt.xticks(df['vocab_size'], labels=[str(v) for v in df['vocab_size']], rotation=45)
        plt.xlabel('Vocab Size (log scale)', fontsize=14, fontweight='bold')
        plt.ylabel(score, fontsize=14, fontweight='bold')
        plt.title(f'{score} Performance of Subword Tokenizers', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)


    # Main plotting logic
    sub_tokenizers = performance_df[performance_df['tokenizer_type'] != 'Atom-wise']['tokenizer_type'].unique()
    vocab_sizes = sorted(performance_df['vocab_size'].unique())
    atomic_score = performance_df[performance_df['tokenizer_type'] == 'Atom-wise'].groupby('mol_rep').max()[score]
    
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    colors = plt.cm.viridis(np.linspace(0, 1, len(sub_tokenizers)))
    
    width = 0.8 / len(sub_tokenizers)  # width to offset points within each vocab size group

    for idx, tokenizer in enumerate(sub_tokenizers):
        for vocab_size in vocab_sizes:
            subset_df = performance_df[(performance_df['tokenizer_type'] == tokenizer) & (performance_df['vocab_size'] == vocab_size)]
            if not subset_df.empty:
                y = subset_df[score].values[0]
                x = vocab_size + (idx - len(sub_tokenizers)/2 + 0.5) * width  # compute x-coordinate within the "box"
                plt.scatter(x, y, label=tokenizer if vocab_size == vocab_sizes[0] else "", color=colors[idx])
                annotate_scores(ax, x, [y], [y])
                
    plot_atomic_scores(ax, atomic_score)
    configure_plot(ax, performance_df, score)
    
    plt.savefig(f'./plots/scores_performance/{score}_subword_tokenizers_vocab_performance{file_name}.png')
    plt.close()





# plot 4: Compare vocabsizes performance on the best subword tokenizer
def determine_best_subword_tokenizer(performance_df, score):
    sub_word_tokenizers = performance_df[performance_df['tokenizer_type'] != 'Atom-wise']
    tokenizer_best_vocabs = performance_df.loc[performance_df.groupby(['tokenizer_type','mol_rep'])[score].idxmax()][['tokenizer_type', 'vocab_size','mol_rep',score, 'task', 'target_data']]
    subword_tokenizers_df = tokenizer_best_vocabs[tokenizer_best_vocabs['tokenizer_type'].isin(sub_word_tokenizers['tokenizer_type'])]
    return subword_tokenizers_df[subword_tokenizers_df[score] == subword_tokenizers_df[score].max()]['tokenizer_type'].values[0]


def plot_best_subword_tokenizer_performance(performance_df, score, file_name, best_subword_tokenizer=None, save_plot=True):
    """
    Plots the performance of the best subword tokenizer across different vocab sizes.
    
    :param performance_df: DataFrame containing the performance data.
    :param score: The score type to be plotted.
    :param file_name: The suffix of the file name where the plot will be saved.
    :param best_subword_tokenizer: Optional parameter to specify the best tokenizer.
    :param save_plot: Boolean, if True the plot will be saved, otherwise displayed.
    """
    def plot_atomic_scores(ax, atomic_score):
        atomic_colors = ['red', 'blue']

        for idx, (mol_rep, atomic_s) in enumerate(atomic_score.iteritems()):
            ax.axhline(y=atomic_s, color=atomic_colors[idx], linestyle='--', alpha=0.5, label=f'{mol_rep} Atomic Tokenizer')
            ax.text(ax.get_xlim()[0], atomic_s + 0.005, f"{atomic_s:.3f}", va="center", ha="right", color=atomic_colors[idx], fontsize=11, fontweight='bold') # shifted the score slightly above
            ax.text(ax.get_xlim()[0] + 0.1, atomic_s, f"{mol_rep}", va="center", ha="right", color=atomic_colors[idx], fontsize=11, fontweight='bold')  # made fontsize a bit bigger and shifted the rep name slightly to the right

        # add legend
        ax.legend(loc='lower right', fontsize=10)



    def plot_error_bars(ax, df, score, color):
        yerr_lower = df[score] - df[f'{score}_lb']
        yerr_upper = df[f'{score}_ub'] - df[score]
        ax.errorbar(df['vocab_size'], df[score], yerr=[yerr_lower.values, yerr_upper.values], fmt='o', markersize=8, color=color, linewidth=2, capsize=5)

    # If the best_subword_tokenizer is not provided, set it as 'BPE'.
    
    if not best_subword_tokenizer:
        best_subword_tokenizer = determine_best_subword_tokenizer(performance_df, score)

    else:
            best_subword_tokenizer = best_subword_tokenizer
    
    logger.info(f"best_subword_tokenizer: {best_subword_tokenizer}")
    best_subword_tokenizer_performance_df = performance_df[performance_df['tokenizer_type'] == best_subword_tokenizer]
    
    # if task is y, change it to HIPS
    if best_subword_tokenizer_performance_df['task'].iloc[0] == 'y':
        best_subword_tokenizer_performance_df['task'] = best_subword_tokenizer_performance_df['task'].apply(lambda x: 'HIPS' if x == 'y' else x)

    


    plt.figure(figsize=(16, 8))
    ax = plt.gca()
    colors = plt.cm.viridis(np.linspace(0, 1, len(best_subword_tokenizer_performance_df[score])))
    sorted_colors = [x for _, x in sorted(zip(best_subword_tokenizer_performance_df[score], colors))]
    
    previous_upper_yerr = None
    for (x, y), color in zip(zip(best_subword_tokenizer_performance_df['vocab_size'], best_subword_tokenizer_performance_df[score]), sorted_colors):
        size = 100 + 15 * (y - min(best_subword_tokenizer_performance_df[score]))
        plt.scatter(x, y, color=color, s=size, edgecolor='black', linewidth=1.5)
        plt.text(x, y + 0.007, f"{y:.3f}", ha="center", va="bottom", fontsize=14, color='black', fontweight='bold')

     

        # Display vocab size clearly on the x-axis
        plt.text(x, ax.get_ylim()[0], f"{x}", ha="center", va="top", fontsize=14, color='black', rotation=45)

    
    # plot_atomic_scores
    atomic_score = performance_df[performance_df['tokenizer_type'] == 'Atom-wise'].groupby('mol_rep').max()[score]
    plot_atomic_scores(ax, atomic_score)
    
            
        

    plot_error_bars(ax, best_subword_tokenizer_performance_df, score, 'gray')
    label= 'AUC ROC'
    # label = 'Perplexity'
    plt.xscale('log')
    plt.xlabel('Vocab Size (Log Scale)', fontsize=14, fontweight='bold')
    plt.ylabel(label, fontsize=14, fontweight='bold')
    plt.title(f'{label} of {best_subword_tokenizer} Tokenizer Across Different Vocab Sizes on Task {best_subword_tokenizer_performance_df["task"].iloc[0]}', fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.6)
    
    norm = mcolors.Normalize(vmin=min(best_subword_tokenizer_performance_df[score]), vmax=max(best_subword_tokenizer_performance_df[score]))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label=label, orientation='vertical')
    
    if save_plot:
        # plt.tight_layout()
        plt.savefig(f'./plots/scores_performance/{score}_best_tokenizer_performance_{best_subword_tokenizer}{file_name}.png')
        plt.close()
    else:
        plt.show()


# plot 5: Effect of vocab size on score
def effect_of_vocab_on_score(df , score , file_name):
    plt.scatter(df['vocab_size'], df[score], marker='o', color='blue', alpha=0.5)
    plt.xlabel('Vocabulary Size')
    plt.ylabel(f'score: {score}')
    plt.title('Effect of Vocab Size on AUC-ROC-Delong')
    plt.grid(True)
    plt.savefig(f'./plots/scores_performance/{score}_effect_of_vocab_on_score{file_name}.png')
    plt.close()

def compare_tokenizers(df, score):
    tokenizers = df['tokenizer_type'].unique()
    avg_scores = []

    for tokenizer in tokenizers:
        avg_score = df[df['tokenizer_type'] == tokenizer][score].max()
        avg_scores.append(avg_score)
    
    # sort the tokenizers based on their average score
    avg_scores, tokenizers = zip(*sorted(zip(avg_scores, tokenizers), reverse=True))
    # Plotting
    plt.bar(tokenizers, avg_scores)
    plt.xlabel('Tokenizer Type')
    plt.ylabel('Average AUC-ROC-Delong')
    plt.title('Comparison between different tokenizers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    logger.info(f"saving plot to: ./plots/scores_performance/compare_tokenizers.png")
    plt.savefig(f'./plots/scores_performance/compare_tokenizers.png')
    plt.close()

# plot 6: Compare tokenizers based on score at their best vocab (VIOLIN PLOT)
def plot_violin(df, score,  filename, legend_location='upper right', samples_column='AUC_ROC_delong_samples'):
    """
    Plots the distribution of scores for different configurations using a violin plot.

    Parameters:
    - df: DataFrame containing the performance data.
    - score: The score type to be plotted.
    - filename: The suffix of the file name where the plot will be saved.
    - legend_location: The location of the legend in the plot.
    """

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 5))

    # Create a label for each row in the dataframe
    labels = [f"{row['tokenizer_type']}_{row['mol_rep']}_{row['vocab_size']}" for _, row in df.iterrows()]
    
    # Extract distribution for each label
    distribution = {label: getattr(row, samples_column) for label, row in zip(labels, df.itertuples())}


    # Define colors for the violins
    colors = plt.cm.viridis(np.linspace(0, 1, len(distribution)))

    # Create a violin plot
    parts = ax.violinplot(distribution.values(), showmeans=True, showmedians=True)

    # Set colors for each violin
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # Set the x-axis labels
    ax.set_xticks(np.arange(1, len(distribution) + 1))
    ax.set_xticklabels(distribution.keys(), rotation=45, ha='right') # added rotation for better readability

    # Set the title and y-axis label
    ax.set_title(f'{score} Distribution of Different Configurations', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{score} score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Configurations', fontsize=12, fontweight='bold')

    # Add horizontal grid lines
    ax.yaxis.grid(True)

    # Add a legend
    # ax.legend(parts['bodies'], distribution.keys(), loc=legend_location)


    # Show median value inside the violin plots
    for pc in parts['bodies']:
        pc.set_alpha(0.5)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_linestyle('--')

    # Save the plot
    plot_path = os.path.join('./plots/scores_performance', f'{filename}_violin_plot.png')
    logger.info(f"Plot saved to {plot_path}")
    plt.tight_layout() # to ensure labels fit well
    plt.savefig(plot_path)
    plt.close()

# plot 7: Compare tokenizers based on score at their best vocab (BOXPLOT)
def plot_boxplot(df,score_column, filename):
    # Extract labels for each boxplot
    labels = df.apply(lambda row: f"{row['tokenizer_type']} {row['mol_rep']} {row['vocab_size']}", axis=1)

    # Extract the data for each boxplot
    data = df[score_column].tolist()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 5))

    # Create the boxplot
    ax.boxplot(data, vert=True, patch_artist=True)

    # Set the x-axis labels and title
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title('Boxplot of Score Distributions', fontsize=16)
    ax.set_ylabel('Score', fontsize=14)

    # Save the plot
    plot_path = os.path.join('./plots/scores_performance', filename)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()



def analyze_perplexity(data, file_name):
    
    # Sort data by train_data_size for proper visualization
    data = data.sort_values(by="train_data_size")

    # Transform perplexity for better visualization
    # data["scaled_perplexity"] = (data["length_perplexity"] - 1) * 1e5
    data["scaled_perplexity"] = data["perplexity"] 
    
    # Plotting
    plt.figure(figsize=(15, 8))
    plt.plot(data["train_data_size"], data["scaled_perplexity"], 
             marker='s', linestyle='-', color='darkcyan', markersize=8, linewidth=2.5)
    
    # Annotating key points for clarity
    for x, y in zip(data["train_data_size"], data["scaled_perplexity"]):
        plt.text(x, y, f"{y:.3f}", ha="left", va="bottom", fontsize=12, color='black', fontweight='bold')
    
    # annotate annotate train data only the last point
    # plt.text(data["train_data_size"].iloc[-1], data["scaled_perplexity"].iloc[-1], f"{data['train_data_size'].iloc[-1]}", ha="left", va="bottom", fontsize=10, color='black')
    plt.xscale("log")
    # plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Training Data Size (log scale)", fontsize=14, fontweight='bold')
    plt.ylabel("Perplexity per Atoms (Symbols)", fontsize=14, fontweight='bold')
    plt.title("Effect of Increasing Training Data Size on Perplexity Peformance", fontsize=16)
    
    # annotate  data size 224K of last point in x scale 
    plt.text(data["train_data_size"].iloc[-1], data["scaled_perplexity"].iloc[-1], f"{data['train_data_size'].iloc[-1]}", ha="left")
  
    
    

    # Adjust grid appearance
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'./plots/pre_train_performance/perplexity_{file_name}.png')
    plt.close()


def plot_max_scores_by_task(csv_file):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)
    # Group by 'task' and find the index of the row with the maximum 'auc_roc_delong'
    idx_max_scores = df.groupby('task')['auc_roc_delong'].idxmax()
    
    # Create a new DataFrame with the rows that correspond to the maximum 'auc_roc_delong' for each task
    max_scores_df = df.loc[idx_max_scores]
    
    # filtered_df : task : score : vocab_size
    filtered_df = df[['task', 'auc_roc_delong', 'auc_roc_delong_lb', 'auc_roc_delong_ub' , 'perplexity',
                       'perplexity_lb', 'perplexity_ub', 'vocab_size',
                       ]]
    filtered_df = filtered_df.sort_values(by='task')
    # add new column with confident interval range
    filtered_df['AUC_variability'] = filtered_df['auc_roc_delong'] - filtered_df['auc_roc_delong_lb']
    filtered_df['perplexity_variability'] = filtered_df['perplexity'] - filtered_df['perplexity_lb']
    # Add variability to origin score column +/-
    filtered_df['auc_roc_delong'] = filtered_df['auc_roc_delong'].apply(lambda x: f"{x:.3f}")
    filtered_df['auc_roc_delong_lb'] = filtered_df['auc_roc_delong_lb'].apply(lambda x: f"{x:.3f}")
    filtered_df['auc_roc_delong_ub'] = filtered_df['auc_roc_delong_ub'].apply(lambda x: f"{x:.3f}")
    filtered_df['AUC_variability'] = filtered_df['AUC_variability'].apply(lambda x: f"{x:.3f}")
    filtered_df['perplexity'] = filtered_df['perplexity'].apply(lambda x: f"{x:.3f}")
    filtered_df['perplexity_lb'] = filtered_df['perplexity_lb'].apply(lambda x: f"{x:.3f}")
    filtered_df['perplexity_ub'] = filtered_df['perplexity_ub'].apply(lambda x: f"{x:.3f}")
    filtered_df['perplexity_variability'] = filtered_df['perplexity_variability'].apply(lambda x: f"{x:.3f}")
    # re arrange columns
    # filtered_df = filtered_df.drop(['auc_roc_delong_lb', 'auc_roc_delong_ub'], axis=1)
    # filtered_df = filtered_df.drop(['perplexity_lb', 'perplexity_ub'], axis=1)

    filtered_df = filtered_df[['task', 'auc_roc_delong', 'auc_roc_delong_lb', 'auc_roc_delong_ub',
                               'AUC_variability', 'perplexity', 'perplexity_lb', 'perplexity_ub',
                               'perplexity_variability', 'vocab_size']]
    filtered_df = filtered_df.reset_index(drop=True)
    
    # Extract tasks with max auc_roc_delong
    auc_df= filtered_df[['task', 'auc_roc_delong', 'auc_roc_delong_lb', 'auc_roc_delong_ub',
                         'AUC_variability', 'vocab_size']]
    max_auc_roc_delong = auc_df.groupby('task')['auc_roc_delong'].max()
    # if duplicate tasks with same max auc_roc_delong, take the one with the max vocab_size
    max_auc_roc_df = pd.merge(max_auc_roc_delong, auc_df, on=['task', 'auc_roc_delong'], how='left')
    # max_auc_roc_df = max_auc_roc_df.sort_values(by='auc_roc_delong', ascending=False)
    max_auc_roc_df = max_auc_roc_df.rename(columns={'vocab_size': 'auc_optimal_vocab'})
    # reset index
    max_auc_roc_df = max_auc_roc_df.reset_index(drop=True)

    # Extract tasks with min perplexity
    perplexity_df= filtered_df[['task', 'perplexity', 'perplexity_lb', 'perplexity_ub',
                                'perplexity_variability' , 'vocab_size']]
    min_perplexity = perplexity_df.groupby('task')['perplexity'].min()
    min_perplexity = min_perplexity.reset_index()
    min_perplexity_df = pd.merge(min_perplexity, perplexity_df, on=['task', 'perplexity'], how='left')
    # min_perplexity_df = min_perplexity_df.sort_values(by='perplexity', ascending=True)
    min_perplexity_df = min_perplexity_df.rename(columns={'vocab_size': 'perplexity_optimal_vocab'})
    # reset index
    min_perplexity_df = min_perplexity_df.reset_index(drop=True)
    # combine the two dataframes
    optimal_scores_df = pd.merge(max_auc_roc_df, min_perplexity_df, on='task', how='left')
    # drop lower and upper bounds columns
    optimal_scores_df = optimal_scores_df.drop(['auc_roc_delong_lb', 'auc_roc_delong_ub', 'perplexity_lb', 'perplexity_ub'], axis=1)
    # change columns order 
    optimal_scores_df = optimal_scores_df[['task', 'auc_roc_delong', 'AUC_variability', 'auc_optimal_vocab', 'perplexity', 'perplexity_variability', 'perplexity_optimal_vocab']]
    optimal_scores_df = optimal_scores_df.reset_index(drop=True)


    # sort dfs by values 
    filtered_df = filtered_df.sort_values(by='task')
    max_auc_roc_df = max_auc_roc_df.sort_values(by='auc_roc_delong', ascending=False)
    min_perplexity_df = min_perplexity_df.sort_values(by='perplexity', ascending=True)
    optimal_scores_df = optimal_scores_df.sort_values(by='task')


    logger.info(f"filtered_df: {filtered_df}")
    logger.info(f"SR-ATAD5: {filtered_df[filtered_df['task'] == 'SR-HSE']}")
    logger.info(f"auc_df: {max_auc_roc_df}")
    logger.info(f"perplexity_df: {min_perplexity_df}")
    logger.info(f"optimal_scores_df: {optimal_scores_df}")


    # Create a scatter plot
    scores_df = max_auc_roc_df
    scores_df = scores_df.reset_index(drop=True)

    # check all values to be float , vocab to be int 
    # scores_df['perplexity'] = scores_df['perplexity'].astype(float)
    # scores_df['perplexity_lb'] = scores_df['perplexity_lb'].astype(float)
    # scores_df['perplexity_ub'] = scores_df['perplexity_ub'].astype(float)
    # scores_df['perplexity_variability'] = scores_df['perplexity_variability'].astype(float)
    # scores_df['perplexity_optimal_vocab'] = scores_df['perplexity_optimal_vocab'].astype(int)
    
    # if two rows has the same score, take the one with the max vocab_size
    scores_df = scores_df.drop_duplicates(subset=['task', 'auc_roc_delong'], keep='first')
    scores_df = scores_df.reset_index(drop=True)
    # logger.info(f"scores_df: {scores_df}")

    scores_df['auc_roc_delong'] = scores_df['auc_roc_delong'].astype(float)
    scores_df['auc_roc_delong_lb'] = scores_df['auc_roc_delong_lb'].astype(float)
    scores_df['auc_roc_delong_ub'] = scores_df['auc_roc_delong_ub'].astype(float)
    scores_df['AUC_variability'] = scores_df['AUC_variability'].astype(float)
    scores_df['auc_optimal_vocab'] = scores_df['auc_optimal_vocab'].astype(int)
    

    # score = 'perplexity'
    # vocabs = scores_df['perplexity_optimal_vocab']
    
    
    score = 'auc_roc_delong'
    vocabs= scores_df['auc_optimal_vocab']
    logger.info(f"scores_df: {scores_df}")
    
    

    # plot scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(scores_df['task'], scores_df[score], 
                c=vocabs, cmap='viridis', s=100, alpha=0.5)
    
    # Annotate the points with the 'vocab_size'
    for i, row in scores_df.iterrows():
        plt.text(row['task'], row[score], f"{row['auc_optimal_vocab']}", fontsize=9, ha='center', va='bottom')
    
    # plot CI erros bars form lb and up bounds 
    yerr_lower = scores_df[score] - scores_df[f'{score}_lb']
    yerr_upper = scores_df[f'{score}_ub'] - scores_df[score]
    plt.errorbar(scores_df['task'], scores_df[score], yerr=[yerr_lower.values, yerr_upper.values], fmt='o', color='gray', alpha=0.5, linewidth=1.5, capsize=5)
    
    
    if score == 'auc_roc_delong':
        ylable = 'AUC ROC' 
    else:
        ylable = score

    xlable = 'Toxicity Endpoint'                
                

    # Add labels and title
    plt.xlabel(xlable, fontsize=14, fontweight='bold')
    plt.ylabel(ylable, fontsize=14, fontweight='bold')
    plt.title('Best performing Task at Optimal Vocab Size')
    plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
    plt.colorbar(label='Vocab Size')  # Show color scale for vocab size
    plt.tight_layout()
    plt.savefig(f'./plots/scores_performance/max_scores_by_task.png')
    plt.close()




def plot_vocab_vs_perplexity(csv_path , file_name):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    # Filter the DataFrame to include only the necessary columns of mol_rep =smiles
    # df = df[df['mol_rep'] == 'smiles']
    df = df[['vocab_size', 'perplexity', 'tokenizer_type', 'mol_rep' , 'task']]
    # chane task from y to HIPS
    if df['task'].iloc[0] == 'y':
        df['task'] = df['task'].apply(lambda x: 'HIPS' if x == 'y' else x)
    # Sort the DataFrame based on vocab_size for clarity in plotting
    df_sorted = df.sort_values(by='vocab_size')
    
    
   
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted['vocab_size'], df_sorted['perplexity'], '+-', color='blue', alpha=0.5)
    
    # annotate the points with the vocab size
    for i, row in df_sorted.iterrows():
        plt.text(row['vocab_size'], row['perplexity'], f"{row['vocab_size']}", fontsize=9, ha='center', va='bottom')

    # draw horizontal line at perplexity of atomic tokenizer
    smiles_atomic_tokenizer_perplexity = df_sorted[df_sorted['tokenizer_type'] == 'Atom-wise']['perplexity'].values[0]
    plt.axhline(y=smiles_atomic_tokenizer_perplexity, color='black', linestyle='--', alpha=0.5, label=f'SMILES Atomic Tokenizer')

    if len(df_sorted[df_sorted['tokenizer_type'] == 'Atom-wise']['perplexity'].values) > 1:
        selfies_atomic_tokenizer_perplexity = df_sorted[df_sorted['tokenizer_type'] == 'Atom-wise']['perplexity'].values[1]
        plt.axhline(y=selfies_atomic_tokenizer_perplexity, color='red', linestyle='--', alpha=0.5, label=f'SELFIES Atomic Tokenizer')
    
    

    plt.xscale('log')  # log scale for x-axis to handle wide range of vocab sizes
    plt.xlabel('Vocabulary Size (log scale)', fontsize=14, fontweight='bold')
    plt.ylabel('Perplexity', fontsize=14, fontweight='bold')
    plt.title(f'Perplexity of BPE Tokenizer Models at Different Vocab Sizes on Task {df_sorted["task"].iloc[0]}')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'./plots/scores_performance/{file_name}_vocab_vs_perplexity.png')
    


if __name__ == "__main__":
    # file = f'auc_scores_batch64_lr0.0001_epochs55_augmentationFalse_resultes.csv' # no aug in both aux and main -> verified
    # file = f'auc_scores_batch64_lr0.0001_epochs55_augmentationTrue_resultes.csv' # aug only in aux and not in main
    # file = f'auc_scores_batch64_lr0.0001_epochs55_tox21_taskSR-p53_augmentationFalse.csv' # no augmentation (MAC frag is the BEST)(Not sure if AUX) verified 

    tox_file = f'auc_scores_batch128_lr0.0001_epochs30_tox21_taskSR-p53_augmentationTrue.csv' # aug both in aux and main  verified
    clintox_file = f'auc_scores_batch128_lr0.0001_epochs30_clintox_taskCT_TOX_augmentationTrue.csv'  # aug both in aux and main  verified
               
    # file_final = f'auc_scores_batch128_lr0.0001_epochs30_taskCT_TOX_aux_taskCT_TOX_AugmentationAuxTrue_resultes.csv' # aug only in aux and not in main    
    # file = f'auc_scores_batch128_lr0.0001_epochs20_taskSR-p53_aux_taskSR-p53_AugmentationAuxTrue_resultes.csv' # aug only in aux and not in main

    # file = f'auc_scores_batch128_lr0.0001_epochs40_tox21_taskSR-p53_augmentationTrue.csv' # aug both in aux and main (NO auc_roc_delong)

    file_name = tox_file
    data = f"./logs/fine_tune/{file_name}"
    data = pd.read_csv(data)
    logger.info(f"length of the dataframe: {len(data)}")
    logger.info(f"columns names of the dataframe: {data.columns}")
    logger.info(f" tokenizers types: {data['tokenizer_type'].unique()}")

    scores = ['auc_roc','auc_prc','f1','balanced_accuracy', 'accuracy','auc_roc_delong']
    scores_sing = ['auc_roc_delong']
    score = 'auc_roc_delong'   
    # Group the results by tokenizer_type and get the maximum AUC for each tokenizer.
    best_scores_performance_df = data.groupby(['tokenizer_type','mol_rep']).max()[score]
    # sort the best scores in descending order
    best_scores_performance_df = best_scores_performance_df.sort_values(ascending=False)
    logger.info(f"best {score} for each tokenizer: {best_scores_performance_df}")
    
    # plot 1: Compare atom-wise tokenizers (SELFIES and SMILES) based on score
    # plot_atom_wise_performance(data, score, file_name)

    # plot 2: Compare tokenizers based on score at their best vocab
    plot_best_vocabs_performance(data, score, file_name)

    # plot 4: Compare vocab_sizes performance on the best subword tokenizer
    plot_best_subword_tokenizer_performance(data, score, file_name, best_subword_tokenizer='BPE', save_plot=True)

    
    # plot 5 : # table results
    # file= f'auc_scores_batch128_lr0.0001_epochs100_BPE_smiles_tox21.csv'
    # file= f'auc_scores_batch128_lr0.0001_epochs40_BPE_smiles_tox21_augTrue.csv'
    # file = f'auc_scores_batch128_lr0.0001_epochs40_BPE_smiles_hips_augFalse.csv'
    file = f'auc_scores_batch128_lr0.0001_epochs100_BPE_smiles_hips_augFalse.csv'
    path = './logs/end_points_fine_tune/'+file
    # plot_max_scores_by_task(path)
    


    # plot 6: pre-training performance
    pre_train_log = f'batch_size_16_lr_0.001_masking_prob_0.8_epochs_20_results.csv'
    pre_train_data = pd.read_csv(f"./logs/pre_train/{pre_train_log}")
    # analyze_perplexity(pre_train_data, pre_train_log)


    # plot 7 : perplexity on tox21 dataset
    Bpe_tox21_SR_p53_no_aug = f'auc_scores_batch128_lr0.001_epochs30_BPE_smiles_tox21_augFalse_SR-p53_1.csv' #--> verified
    # Bpe_tox21_SR_p53_aug = f'auc_scores_batch128_lr0.001_epochs30_BPE_smiles_tox21_augTrue_SR-p53.csv'
    # auc_scores_batch128_lr0.0001_epochs30_BPE_smiles_tox21_augFalse_['SR-p53']
    
    
    # file = f'auc_scores_batch128_lr0.0001_epochs50_BPE_smiles_tox21_augFalse_SR-p53.csv'


    # file = f'auc_scores_batch128_lr0.0001_epochs30_smiles_clintox_augFalse_CT_TOX.csv'
    # file = f'auc_scores_batch128_lr0.0001_epochs30_taskCT_TOX_aux_taskCT_TOX_AugmentationAuxTrue_resultes.csv'

    # file = f'auc_scores_batch128_lr0.0001_epochs30_BPE_smiles_tox21_augFalse_SR-p53.csv'

    # file = f'auc_scores_batch128_lr0.0001_epochs30_smiles_tox21_augTrue_SR-p53_1.csv'  #---> verified
    # file =f'auc_scores_batch128_lr0.0001_epochs30_smiles_hips_augFalse_y.csv' #--> verified 
    
    # END POINTS
    # file =f'auc_scores_batch128_lr0.0001_epochs40_smiles_tox21_end_points_augTrue.csv'
    # file =f'auc_scores_batch128_lr0.0001_epochs40_smiles_tox21_end_points_augFalse.csv'
    
    
    path = f'./logs/end_points_fine_tune'
    # name file without csv 
    file_name = file.split('.')[0]
    perlexity_df= pd.read_csv(path+'/'+file)
    # plot_max_scores_by_task(path+'/'+file)

    # score = 'perplexity'  
    # score= 'smoothed_perplexity'
    score = 'auc_roc_delong'
    # plot_vocab_vs_perplexity(path+'/'+file , file_name = file_name)
    # plot_best_subword_tokenizer_performance(perlexity_df, score, file_name = file_name, best_subword_tokenizer='BPE', save_plot=True)
    
    


    
    
    
    # # plot 3.1: Compare vocab_sizes performance for  subword tokenizers
    # plot_vocabulary_performance(data, score, file_name)
    
    
    # # plot 3.2: Compare vocab_sizes performance for  subword tokenizers
    # plot_vocabulary_performance_pinpoints(performance_df, score, file_name)

    
    # plot 5: Effect of vocab size on score
    # effect_of_vocab_on_score(data, score, file_name)

        

    