import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
from scipy import stats
import logging
from sklearn import preprocessing
import pandas as pd
import ast
logger = logging.getLogger(__name__)

# PLOTING FUNCTIONS
def normalize_scores(scores):
    """Normalize the given scores to the range [0, 1] using absolute values.
    
    Args:
        scores (list): List of scores to normalize.
    
    Returns:
        list: Normalized scores.
    """
    min_val, max_val = min(scores), max(scores)
    normalized_scores = [(val - min_val) / (max_val - min_val) for val in scores]
    return normalized_scores


######################################################################################### EXPLAINABILITY PLOTS #############################################################################################################

def plot_bars(values, labels, title, filename, molecule , target_class):
    plt.rcParams["figure.figsize"] = (20,10)
    
    # If values or labels are nested within a list, extract the first element
    if isinstance(values[0], (list, np.ndarray)):
        values = values[0]
    if isinstance(labels[0], (list, np.ndarray)):
        labels = labels[0]
    
    # Normalize the absolute values of the bars to the range [0.5, 1]
    normalized_values = 0.5 + 0.5 * np.abs(values) / max(np.abs(values))
    
    # Create a color list based on the values of the bars
    colors = [(val, 1-val, 0) if v > 0 else (1-val, val, 0) for v, val in zip(values, normalized_values)]
    
    # Plot the bars with the specified colors
    plt.barh(range(len(labels)), values, color=colors)
    plt.yticks(range(len(values)), labels, fontsize=14, fontweight='bold')
    plt.title(f'{title} of MOL:  {molecule}  (Toxicity Class {target_class})', fontsize=20, fontweight='bold')
    plt.xlabel('Importance Value', fontsize=16, fontweight='bold', color='black')
    plt.ylabel('Molecule Fragments', fontsize=16, fontweight='bold', color='black')
    plt.axvline(x=0, color='black', linestyle='--') # Add a vertical line at x=0
    plt.tight_layout()
    

    # Write the values on top of the bars
    for i, v in enumerate(values):
        plt.text(v, i, f"{v:.3f}", va='center', ha='left', fontsize=12, color='black', fontweight='bold')
    
    # Save plot to file
    logger.info(f'Plot saved to {filename}')
    plot_path = os.path.join('./plots/xai/explain', filename)
    plt.savefig(plot_path)
    plt.close()


def plot_combined_heatmap(Explanations , file_name,molecule ,target_class):
    """
    Plots a heatmap comparing different explanation approaches.

    Explanations: Dictionary with approach names as keys and tuple of (values, labels) as values.
    Structure : Explanations = {method: {'scores': [], 'labels': []} for method in ['attention_max', 'attention_avg', 'IG', 'shapley', 'LIME']}
    """
    # Find the union of all labels across all approaches
    all_labels = set()
    for (method, data) in Explanations.items():
        all_labels.update(data['labels'][0])
    all_labels = list(sorted(all_labels))
    
    
    # Initialize a 2D array to hold the values for all approaches and labels
    values_array = np.zeros((len(Explanations), len(all_labels)))
    
    # Fill in the values array
    for i, (method, data) in enumerate(Explanations.items()):
        for label, value in zip(data['labels'][0], data['scores'][0]):
            j = all_labels.index(label)  # Find the column index for this label
            values_array[i, j] = value
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(20, len(Explanations) * 3))
    cax = ax.imshow(values_array, cmap='viridis', aspect='auto')
    
    # Customize the plot
    ax.set_yticks(np.arange(len(Explanations)))
    ax.set_yticklabels(Explanations.keys(), fontsize=12)  # Increase font size
    ax.set_xticks(np.arange(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=12)  # Increase font size
    
    # Add values to the cells
    for i in range(len(Explanations)):
        for j in range(len(all_labels)):
            ax.text(j, i, f'{values_array[i, j]:.2f}', ha='center', va='center', color='w', fontsize=10)
    
    # Add a title
    ax.set_title(f'Explanations Across Different Approaches for    {molecule}    (Toxicity class {target_class})', fontsize=14)
    
    # Add a color bar
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel('Values', rotation=90, fontsize=12)  # Increase font size
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot
    # plt.show()

    # Save the plot
    plt.savefig('./plots/xai/explain/' + file_name)
    plt.close()





############################################################################################################
def plot_values(values, labels, title, filename):
    """Plot the given values with labels and save the plot with the given filename.
    
    Args:
        values (list): List of values to plot.
        labels (list): List of labels for the values.
        title (str): Title of the plot.
        filename (str): Filename to save the plot.
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Normalize values for color intensity
    min_val, max_val = np.min(values), np.max(values)
    normalized_values = (values - min_val) / (max_val - min_val)
    
    for i, value in enumerate(values):
        if value > 0:  # toxic
            color = (1, 1 - normalized_values[i], 1 - normalized_values[i])  # Red with intensity based on value
        else:  # non-toxic
            color = (1 - normalized_values[i], 1, 1 - normalized_values[i])  # Green with intensity based on value
        
        # Create bars and set color
        bar = ax.bar(i, value, color=color)
        
        # Display value on top of the bar
        ax.text(i, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    
    # Set x-axis labels and title
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_title(title)
    
    # Add gridlines
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Ensure layout fits without overlapping labels
    plt.tight_layout()
    
    # Save plot to file
    plot_path = os.path.join('./plots/explain', filename)
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")
    plt.close()


################### EVALUATION PLOTS ######################################################################################
def plot_stability(similarity_metrics, filename):
    """
    Generate a bar plot of average Jaccard similarity and standard deviation for each XAI method.
    
    Parameters:
    - similarity_metrics (dict): Dictionary containing average and standard deviation of Jaccard similarities for each method.
    """
    
    methods = list(similarity_metrics.keys())
    avg_vals = [similarity_metrics[method][0] for method in methods]
    std_vals = [similarity_metrics[method][1] for method in methods]

    x_pos = np.arange(len(methods))

    plt.figure(figsize=(12, 7))
    bars = plt.bar(x_pos, avg_vals, yerr=std_vals, align='center', alpha=0.7, ecolor='black', capsize=10, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
    plt.xlabel('XAI Methods', fontsize=14)
    plt.ylabel(' Stability Score', fontsize=14)
    plt.title('Stability of Explanations', fontsize=16)
    plt.xticks(x_pos, methods, rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    
    # Add bar value on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom', fontsize=10)

    # Save plot to file
    plot_path = os.path.join('./plots/xai/eval', filename)
    plt.savefig(plot_path)
    logger.info(f'Plot saved to {plot_path}')
    plt.close()

def plot_violin(distribution, filename, metric, legend_location='lower right'):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.set_xticklabels(distribution.keys())

    # Set the title and y-axis label
    ax.set_title(f'{metric} Distribution of Different XAI Methods', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{metric} score', fontsize=12, fontweight='bold')
    ax.set_xlabel('XAI Method', fontsize=12, fontweight='bold')

    # Add horizontal grid lines
    ax.yaxis.grid(True)

    # Add a legend
    ax.legend(parts['bodies'], distribution.keys(), loc=legend_location)

    # show median value inside the violin plots
    for pc in parts['bodies']:
        pc.set_alpha(0.5)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_linestyle('--')

    # Save the plot
    plot_path = os.path.join('./plots/xai/eval', filename)
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")
    # Uncomment the following line if you have a logger
    # logger.info(f"Plot saved to {plot_path}")
    plt.close()


def plot_XAI_scores(XAI_scores, filename , metric):
    # Ensure the directory exists
    directory = './plots/xai/eval'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Extracting names and scores
    methods = list(XAI_scores.keys())
    scores = list(XAI_scores.values())
    
    # Use a diverging colormap
    cmap = plt.get_cmap('RdYlGn_r')
    colors = [cmap(score) for score in scores]
    # Plotting
    plt.figure(figsize=(12,7))
    bars = plt.bar(methods, scores, color=colors, edgecolor='black', linewidth=0.7)
    plt.xlabel('XAI Methods', fontsize=13)
    plt.ylabel(f'{metric} score', fontsize=13)
    plt.title(f'{metric} Scores of Different XAI Methods', fontsize=14)
    plt.axhline(0, color='black', linewidth=0.8)  # Add a horizontal line at y=0    
    # Adding the text labels inside the bars
    for bar, score in zip(bars, scores):
        y = bar.get_height()
        ha = 'center'
        va = 'bottom' if score >= 0 else 'top'
        color = 'black' if np.abs(score) < 0.01 else 'white'
        plt.text(bar.get_x() + bar.get_width() / 2, y, f"{score:.3f}", ha=ha, va=va, color=color, fontsize=10)
        # write the values on top of the bars
        for i, v in enumerate(scores):
            plt.text(i, v, f"{v:.3f}", va='center', ha='center', fontsize=10)

    # Save plot to file
    plot_path = os.path.join('./plots/xai/eval', filename)
    plt.savefig(plot_path)
    logger.info(f'Plot saved to {plot_path}')
    plt.close()
    
    
#########################################  report   analyze CSV results    #########################################################

# Model preference plots
def choose_best_model(csv, tox_weight=0.6, nontox_weight=0.4):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv)
    # Ensure weights sum to 1 for proper normalization
    if tox_weight + nontox_weight != 1:
        raise ValueError("Weights must sum to 1.")
    
    # Calculate the score for each task
    data['tox_correctness_score'] = data['toxic_mols_predicted_toxic'] / data['tox_mols']
    data['nontox_correctness_score'] = data['non_toxic_mols_predicted_non_toxic'] / data['non_tox_mols']
    
    # Overall score is a weighted average of both scores
    data['overall_score'] = (data['tox_correctness_score'] * tox_weight +
                             data['nontox_correctness_score'] * nontox_weight)
    
    
    # Choose the model with the highest overall score
    best_model = data.loc[data['overall_score'].idxmax()]

    # sub_data = data['task','vocab','tox_mols','toxic_mols_predicted_toxic','non_tox_mols','non_
    #    ...: toxic_mols_predicted_non_toxic','overall_score']

    sub_data = data[['task','vocab','tox_mols','toxic_mols_predicted_toxic','non_tox_mols','non_toxic_mols_predicted_non_toxic','overall_score']]
    return best_model , data



# [1] FAITHFULNESS PLOTS
def report_faithfulness_scores(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Prepare a list to hold the results
    results = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        method = row['method']
        faithfulness_avg = row['faithfullness_avg']
        # Convert the string representation of the list into an actual list
        faithfulness_distribution = ast.literal_eval(row['faithfullness_distribution'])
        # normalize the faithfulness distribution for each method
        # faithfulness_distribution = preprocessing.normalize([faithfulness_distribution], norm='l2')[0]
        

        # marginal error fron standard error of the mean
        # marginal_error = np.std(faithfulness_distribution) / np.sqrt(len(faithfulness_distribution))
        # 95 % confidence interval
        marginal_error = 1.96 * np.std(faithfulness_distribution) / np.sqrt(len(faithfulness_distribution))
        
        # Store the results
        results.append({
            'Method': method,
            'Faithfulness Score': faithfulness_avg,
            'Marginal Error': marginal_error,
            'Faithfulness Scores': faithfulness_distribution
        })
    
    # Convert the results into a DataFrame for reporting
    results_df = pd.DataFrame(results)
    # sckiit normalize preprocessing.normalize
    # results_df['Faithfulness Score'] = preprocessing.normalize([results_df['Faithfulness Score']], norm='l2')[0]
    # sort the results by the faithfulness score
    results_df = results_df.sort_values(by='Faithfulness Score', ascending=False)
    # Draw the faithfulness bars plot
    plot_faithfulness_bars(results_df)

    
    #remove the faithfulness distribution column
    results_df = results_df[['Method', 'Faithfulness Score', 'Marginal Error']]
    logger.info(f'Faithfulness scores:\n{results_df}')



    return results_df

def plot_faithfulness_bars(results_df):

    
    # Set the figure size and style for better readability
    plt.figure(figsize=(12, 8))
    plt.style.use('ggplot')

    # Extracting data for plotting
    methods = results_df['Method']
    # Change shapley to SHAP in xlabels
    methods = [method.replace('shapley', 'SHAP') for method in methods]
    avg_faithfulness_scores = results_df['Faithfulness Score']
    marginal_errors = results_df['Marginal Error']

    # Generate a color map for visual variety
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))

    # Creating bars with error bars
    bars = plt.bar(methods, avg_faithfulness_scores, yerr=marginal_errors, 
                   align='center', alpha=0.7, ecolor='black', capsize=10, 
                   color=colors)

    # Adding labels and title
    plt.xlabel('XAI Methods', fontsize=14, fontweight='bold', color='black')
    plt.ylabel('Faithfulness Score', fontsize=14, fontweight='bold', color='black')
    plt.title('Faithfulness (Fidelity) of Explanations Across XAI Methods', fontsize=16)

    # Rotate x-axis labels for clarity
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Annotate bars with their respective values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                 f'{yval:.2f}', ha='left', va='bottom', fontsize=12, color='black', fontweight='bold')
        
    #add sign on the corner the more value the better (arrow up) top right corner
    plt.annotate('↑', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=20, ha='right', va='top')

    

    # Adjust layout for a clean look
    plt.tight_layout()

    # Save plot to file
    plot_path = './plots/xai/eval/faithfulness_bars.png'
    plt.savefig(plot_path)
    print(f'Plot saved to {plot_path}')  # Replace with logger if needed
    plt.close()
    



# [2] POST-HOC EXPLANATION STABILITY PLOTS
def report_local_lipschitz_scores(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Prepare a list to hold the results
    results = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        method = row['method']
        local_lipschitz_constant = row['local_lipschitz_constant']
        # Convert the string representation of the list into an actual list
        local_lipschitz_distribution = ast.literal_eval(row['local_lipschitz_constant_distribution'])
        # normalize the local_lipschitz_distribution for each method
        # local_lipschitz_distribution = preprocessing.normalize([local_lipschitz_distribution], norm='l2')[0]

        
       
        # marginal error fron standard error of the mean 95 % confidence interval
        marginal_error = 1.96 * np.std(local_lipschitz_distribution) / np.sqrt(len(local_lipschitz_distribution))
        
        # Store the results
        results.append({
            'Method': method,
            'Local Lipschitz Constant': local_lipschitz_constant,
            'Marginal Error': marginal_error,
            'Local Lipschitz Constant Distribution': local_lipschitz_distribution
        })
    
    # Convert the results into a DataFrame for reporting
    results_df = pd.DataFrame(results)
    # sckiit normalize preprocessing.normalize
    # results_df['Local Lipschitz Constant'] = preprocessing.normalize([results_df['Local Lipschitz Constant']], norm='l2')[0]
    # sort the results by Local Lipschitz Constant
    results_df = results_df.sort_values(by='Local Lipschitz Constant', ascending=False)
    # Draw the Local Lipschitz Constant bars plot
    plot_local_lipschitz_bars(results_df)
    

    # remove the Local Lipschitz Constant distribution column
    results_df = results_df[['Method', 'Local Lipschitz Constant', 'Marginal Error']]
    logger.info(f'Local Lipschitz Constant scores:\n{results_df}')
    return results_df


def plot_local_lipschitz_bars(results_df):
    # Extracting the data for the box plot
    
    methods = results_df['Method'].tolist()
    # Change shapley to SHAP in xlabels
    methods = [method.replace('shapley', 'SHAP') for method in methods]
    local_lipschitz_scores = results_df['Local Lipschitz Constant'].tolist()
    marginal_errors = results_df['Marginal Error'].tolist()
    # Plotting bars with error bars
    plt.figure(figsize=(12, 7))
    bars = plt.bar(methods, local_lipschitz_scores, yerr=marginal_errors, align='center', alpha=0.7, ecolor='black', capsize=10, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
    plt.xlabel('XAI Methods', fontsize=14, fontweight='bold', color='black')
    plt.ylabel('RIS Score', fontsize=14, fontweight='bold', color='black')
    plt.title('Relative Input Stability (RIS) of Explanations Across XAI Methods', fontsize=16)
    plt.xticks(methods, rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    # Add bar value on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                 f'{yval:.2f}', ha='left', va='bottom', fontsize=12, color='black', fontweight='bold')

    # add sign on the corner the less value the better (arrow down) top right corner
    plt.annotate('↓', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=20, ha='right', va='top')

    

    # Save plot to file
    plot_path = os.path.join('./plots/xai/eval', 'local_lipschitz_bars.png')
    plt.savefig(plot_path)
    logger.info(f'Plot saved to {plot_path}')
    plt.close()


#[3] Retiration stability plots
def plot_retiration_stability_bars(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    # remove last column
    df = df.iloc[:, :-1]
    # change shapley to SHAP
    df = df.rename(columns={'shapley': 'SHAP'})

    logger.info(f'Retiration stability scores:\n{df}')
    # Extracting the data for the box plot
    methods = df.columns.tolist()
    retiration_stability_scores = df.iloc[0].tolist()
    marginal_errors = df.iloc[1].tolist()
    # Plotting bars with error bars
    plt.figure(figsize=(12, 7))
    bars = plt.bar(methods, retiration_stability_scores, yerr=marginal_errors, align='center', alpha=0.7, ecolor='black', capsize=10, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
    plt.xlabel('XAI Methods', fontsize=14, fontweight='bold', color='black')
    plt.ylabel('Stability Score', fontsize=14, fontweight='bold', color='black')
    plt.title('Reproducible Statbility of Explanations Across XAI Methods', fontsize=16)
    plt.xticks(methods, rotation=45, ha='right', fontsize=12)
    plt.tight_layout()



    # Add bar value on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                 f'{yval:.2f}', ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

    # add sign on the corner the more value the better (arrow up) top right corner
    plt.annotate('↑', xy=(0.99, 0.95), xycoords='axes fraction', fontsize=20, ha='right', va='top')

    # Save plot to file
    plot_path = os.path.join('./plots/xai/eval', 'retiration_stability_bars.png')
    plt.savefig(plot_path)
    logger.info(f'Plot saved to {plot_path}')
    plt.close()


    
    


# [4] FAIRNESS PLOTS
def plot_fairness_comparison(csv_path, filename=f'fairness_comparison.png'):
    # Read the CSV file into a DataFrame
    df = analyze_fairness_results(csv_path)
    # Set the width of the bars
    bar_width = 0.3
    # Set positions of the bars
    index = np.arange(len(df['method']))
    # change shapley to SHAP in df['Method']
    df['method'] = [method.replace('shapley', 'SHAP') for method in df['method']]
    fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    # Plot the first metric comparison (faithfulness)
    ax[0].bar(index, df['toxic_samples_faithfulness'], bar_width, label='Toxic Samples Faithfulness (Minority)', alpha=0.8)
    ax[0].bar(index + bar_width, df['non_toxic_samples_faithfulness'], bar_width, label='Non Toxic Samples Faithfulness (Majority)', alpha=0.8)
    # add marginal error for each bar
    ax[0].errorbar(index, df['toxic_samples_faithfulness'], yerr=df['toxic_samples_faithfulness_marginal_error'], fmt='none', ecolor='black', capsize=3)
    ax[0].errorbar(index + bar_width, df['non_toxic_samples_faithfulness'], yerr=df['non_toxic_samples_faithfulness_marginal_error'], fmt='none', ecolor='black', capsize=3)
    
    # draw arrow the higher the better
    ax[0].annotate('↑', xy=(0.99, 0.50), xycoords='axes fraction', fontsize=20, ha='right', va='top')
    
    # Set labels and titles
    ax[0].set_xlabel('XAI Methods', fontsize=14, fontweight='bold')
    ax[0].set_ylabel('Faithfulness Score', fontsize=14, fontweight='bold')
    ax[0].set_title('Faithfulness Comparison between Tox and Non-Tox Subgroups', fontsize=16)
    ax[0].set_xticks(index + bar_width / 2)
    ax[0].set_xticklabels(df['method'], rotation=45, ha='right', fontsize=12)
    ax[0].legend()



    # Plot the second metric comparison (stability)
    ax[1].bar(index, df['toxic_samples_instability'], bar_width, label='Toxic Samples RIS (Minority)', alpha=0.8)
    ax[1].bar(index + bar_width, df['non_toxic_samples_instability'], bar_width, label='Non Toxic Samples RIS (Majority)', alpha=0.8)
    # add marginal error for each bar
    ax[1].errorbar(index, df['toxic_samples_instability'], yerr=df['toxic_samples_instability_marginal_error'], fmt='none', ecolor='black', capsize=3)
    ax[1].errorbar(index + bar_width, df['non_toxic_samples_instability'], yerr=df['non_toxic_samples_instability_marginal_error'], fmt='none', ecolor='black', capsize=3)
    
    # draw arrow the lower the better
    ax[1].annotate('↓', xy=(0.99, 0.50), xycoords='axes fraction', fontsize=20, ha='right', va='top')
    
    # Set labels and titles
    ax[1].set_xlabel('XAI Methods', fontsize=14, fontweight='bold')
    ax[1].set_ylabel('RIS Score',   fontsize=14, fontweight='bold')
    ax[1].set_title('Relative Input Stability (RIS) Comparison between Tox and Non-Tox Subgroups', fontsize=16)
    ax[1].set_xticks(index + bar_width / 2)
    ax[1].set_xticklabels(df['method'], rotation=45, ha='right', fontsize=12)
    ax[1].legend()

    # increase space between subplots
    plt.subplots_adjust(hspace=0.5)

    # increase space between x-ticks and x-labels
    plt.tick_params(axis='x', which='major', pad=20)

    # remove upper part of the plot frame ax[0]
    # ax[0].spines['top'].set_visible(False)
    

    # Adjust layout for readability
    plt.tight_layout()

    # Save plot to file
    plot_path = os.path.join('./plots/xai/eval', filename)
    plt.savefig(plot_path)
    logger.info(f'Plot saved to {plot_path}')
    plt.close()



def analyze_fairness_results(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
        

    fairness_data = {
        'method': [],
        'toxic_samples_faithfulness': [],
        'toxic_samples_faithfulness_marginal_error': [],
        'faithfulness_scores_tox': [],

        'non_toxic_samples_faithfulness': [],
        'non_toxic_samples_faithfulness_marginal_error': [],
        'faithfulness_scores_non_tox': [],

        'toxic_samples_instability': [],
        'toxic_samples_instability_marginal_error': [],
        'instatnility_scores_tox': [],

        'non_toxic_samples_instability': [],
        'non_toxic_samples_instability_marginal_error': [],
        'instatnility_scores_non_tox': [],

        'faithfulness_fairness': [],
        'faithfulness_fairness_marginal_error': [],
        'faithfulness_fairness_scores': [],
        'instability_fairness': [],
        'instability_fairness_marginal_error': [],
        'instability_fairness_scores': [],


    }
    
    # Loop over each row in the DataFrame
    for index, row in df.iterrows():
        # Parse the fairness string into a dictionary
        fairness_dict = ast.literal_eval(row['fairness'])
        
        # Append the data to the fairness_data dictionary
        fairness_data['method'].append(row['method'])
        fairness_data['toxic_samples_faithfulness'].append(fairness_dict['toxic_samples_faithfulness'])
        fairness_data['toxic_samples_faithfulness_marginal_error'].append(fairness_dict['toxic_samples_faithfulness_marginal_error'])
        fairness_data['faithfulness_scores_tox'].append(fairness_dict['faithfulness_scores_tox'])
        fairness_data['non_toxic_samples_faithfulness'].append(fairness_dict['non_toxic_samples_faithfulness'])
        fairness_data['non_toxic_samples_faithfulness_marginal_error'].append(fairness_dict['non_toxic_samples_faithfulness_marginal_error'])
        fairness_data['faithfulness_scores_non_tox'].append(fairness_dict['faithfulness_scores_non_tox'])
        fairness_data['toxic_samples_instability'].append(fairness_dict['toxic_samples_instability'])
        fairness_data['toxic_samples_instability_marginal_error'].append(fairness_dict['toxic_samples_instability_marginal_error'])
        fairness_data['instatnility_scores_tox'].append(fairness_dict['instatnility_scores_tox'])
        fairness_data['non_toxic_samples_instability'].append(fairness_dict['non_toxic_samples_instability'])
        fairness_data['non_toxic_samples_instability_marginal_error'].append(fairness_dict['non_toxic_samples_instability_marginal_error'])
        fairness_data['instatnility_scores_non_tox'].append(fairness_dict['instatnility_scores_non_tox'])
        fairness_data['faithfulness_fairness'].append(fairness_dict['faithfulness_fairness'])
        fairness_data['faithfulness_fairness_marginal_error'].append(fairness_dict['faithfulness_fairness_marginal_error'])
        fairness_data['faithfulness_fairness_scores'].append(fairness_dict['faithfulness_fairness_scores'])
        fairness_data['instability_fairness'].append(fairness_dict['instability_fairness'])
        fairness_data['instability_fairness_marginal_error'].append(fairness_dict['instability_fairness_marginal_error'])
        fairness_data['instability_fairness_scores'].append(fairness_dict['instability_fairness_scores'])


        
    
     
    

    
    # Convert the fairness_data dictionary into a DataFrame for easier analysis
    fairness_df = pd.DataFrame(fairness_data)

    
    # apply statistical test to compare the faithfulness scores of toxic and non toxic samples means for each method
    # t-test for independent samples
    # H0: the means of the two groups are equal
    # H1: the means of the two groups are not equal
    # alpha = 0.05
    # if p-value < alpha: reject H0
    # if p-value > alpha: fail to reject H0
    methods = fairness_df['method'].tolist()
    for method in methods:
        tox_faithfulness_scores = fairness_df.loc[fairness_df['method'] == method, 'faithfulness_scores_tox'].tolist()[0]
        non_tox_faithfulness_scores = fairness_df.loc[fairness_df['method'] == method, 'faithfulness_scores_non_tox'].tolist()[0]
        # t-test for independent samples
        t_statistic, p_value = stats.ttest_ind(tox_faithfulness_scores, non_tox_faithfulness_scores, equal_var=False)
        # logger.info(f'Faithfulness scores for method {method}:\nToxic samples: {tox_faithfulness_scores}\nNon toxic samples: {non_tox_faithfulness_scores}\nt-statistic: {t_statistic}\np-value: {p_value}')
        # if p-value < alpha: reject H0
        # if p-value > alpha: fail to reject H0
        if p_value < 0.05:
            logger.info(f'Faithfulness scores for method {method} are significantly different for toxic and non toxic samples')
        else:
            logger.info(f'Faithfulness scores for method {method} are not significantly different for toxic and non toxic samples')
    
    # add p_value to the dataframe
    fairness_df['faithfulness_p_value'] = [stats.ttest_ind(fairness_df.loc[fairness_df['method'] == method, 'faithfulness_scores_tox'].tolist()[0], fairness_df.loc[fairness_df['method'] == method, 'faithfulness_scores_non_tox'].tolist()[0], equal_var=False)[1] for method in methods]
    
    # apply statistical test to compare the instability scores of toxic and non toxic samples means for each method
    # t-test for independent samples
    # H0: the means of the two groups are equal
    # H1: the means of the two groups are not equal
    # alpha = 0.05
    # if p-value < alpha: reject H0
    # if p-value > alpha: fail to reject H0
    methods = fairness_df['method'].tolist()
    for method in methods:
        tox_instability_scores = fairness_df.loc[fairness_df['method'] == method, 'instatnility_scores_tox'].tolist()[0]
        non_tox_instability_scores = fairness_df.loc[fairness_df['method'] == method, 'instatnility_scores_non_tox'].tolist()[0]
        # t-test for independent samples
        t_statistic, p_value = stats.ttest_ind(tox_instability_scores, non_tox_instability_scores, equal_var=False)
        # logger.info(f'Instability scores for method {method}:\nToxic samples: {tox_instability_scores}\nNon toxic samples: {non_tox_instability_scores}\nt-statistic: {t_statistic}\np-value: {p_value}')
        # if p-value < alpha: reject H0
        # if p-value > alpha: fail to reject H0
        if p_value < 0.05:
            logger.info(f'Instability scores for method {method} are significantly different for toxic and non toxic samples')
        else:
            logger.info(f'Instability scores for method {method} are not significantly different for toxic and non toxic samples')

    # add p_value to the dataframe
    fairness_df['instability_p_value'] = [stats.ttest_ind(fairness_df.loc[fairness_df['method'] == method, 'instatnility_scores_tox'].tolist()[0], fairness_df.loc[fairness_df['method'] == method, 'instatnility_scores_non_tox'].tolist()[0], equal_var=False)[1] for method in methods]





    # remove non_toxic_samples_faithfulness and non_toxic_samples_stability
    fairness_scores = fairness_df.copy()
    fairness_scores = fairness_scores[['method','faithfulness_fairness','instability_fairness','faithfulness_p_value','instability_p_value']]
                                    

    toxic_scores = fairness_df.copy()
    toxic_scores = toxic_scores[['method','toxic_samples_faithfulness','toxic_samples_instability','toxic_samples_instability_marginal_error']]

    non_toxic_scores = fairness_df.copy()
    non_toxic_scores = non_toxic_scores[['method','non_toxic_samples_faithfulness','non_toxic_samples_instability']]

    
    


    # normalized = (x-min(x))/(max(x)-min(x))
    # toxic_scpres['faithfulness_fairness'] = (toxic_scpres['faithfulness_fairness'] - toxic_scpres['faithfulness_fairness'].min()) / (toxic_scpres['faithfulness_fairness'].max() - toxic_scpres['faithfulness_fairness'].min())

    #sckiit normalize preprocessing.normalize
    # fairness_scores['faithfulness_fairness'] = preprocessing.normalize([fairness_scores['faithfulness_fairness']], norm='l2')[0]
    # fairness_scores['instability_fairness'] = preprocessing.normalize([fairness_scores['instability_fairness']], norm='l2')[0]
    fairness_scores['avg_fairness'] = (fairness_scores['faithfulness_fairness'] + fairness_scores['instability_fairness']) / 2
    # faithfulness fairness score by difference between toxic and non toxic faithfulness scores for the same method
    
    
    fairness_scores = fairness_scores.sort_values(by='avg_fairness', ascending=True)

    # toxic_scpres['toxic_samples_faithfulness'] = preprocessing.normalize([toxic_scpres['toxic_samples_faithfulness']], norm='l2')[0]
    # toxic_scpres['toxic_samples_instability'] = preprocessing.normalize([toxic_scpres['toxic_samples_instability']], norm='l2')[0]
    toxic_scores = toxic_scores.sort_values(by='toxic_samples_faithfulness', ascending=True)

    # non_toxic_scpres['non_toxic_samples_faithfulness'] = preprocessing.normalize([non_toxic_scpres['non_toxic_samples_faithfulness']], norm='l2')[0]
    # non_toxic_scpres['non_toxic_samples_instability'] = preprocessing.normalize([non_toxic_scpres['non_toxic_samples_instability']], norm='l2')[0]
    non_toxic_scores = non_toxic_scores.sort_values(by='non_toxic_samples_faithfulness', ascending=True)
    
    
    logger.info(f'Fairness scores:\n{fairness_scores}')
    logger.info(f'Toxic samples scores:\n{toxic_scores}')
    logger.info(f'Non toxic samples scores:\n{non_toxic_scores}')
    
    
    return fairness_df


