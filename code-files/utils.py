from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from scipy.stats import sem, t
from sklearn.metrics import log_loss
import numpy as np
import selfies as sf
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from scaffold_preprocess import Scaffoldprocessor
from delong_ci import calc_auc_ci
import logging
# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os
# import seaborn as sns
import random
from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers




def is_supported_smiles(smile):
        try:
            sf.encoder(smile)
            return True
        except Exception:
            print("Unsupported SMILES: %s" % smile)
            return False
        

def build_vocab(input, output ,special_tokens):
    vocab = set()
    # read fragmented molecules from file
    with open(input, 'r') as f:
        for line in f:
            # Lines separated by comma (',')
            fragments = line.strip().split(',')
            # for fragment in fragments:
            #         vocab.add(fragment.strip())
            vocab.update(fragments)
    # some default tokens from huggingface
    default_toks = special_tokens
    # add default tokens to head of vocab
    vocab = list(default_toks) + list(vocab)
    # Save vocabulary to file
    with open(output, 'w') as f:
        # Write each fragment in a new line
        for fragment in vocab:
            f.write(fragment + '\n')

def count_tokens(filename):
    tokens = []
    token_counts = {}
    with open(filename, 'r') as file:
        for line in file:
            line_tokens = line.strip().split(',')
            tokens.extend(line_tokens)
            for token in line_tokens:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1
    unique_tokens = list(token_counts.keys())
    return len(tokens), len(unique_tokens)

def read_segmented_mols(file_path):
    with open(file_path, "r") as f:
         text_data = f.read().strip().split("\n")

    return text_data
            
def smiles_to_text(dataframe, output_file):
    # write smiles string representation in dataframe to text file
    with open(output_file, 'w') as f:
        for smile in dataframe['smiles']:
            f.write("%s\n" % smile)

def selfies_to_text(dataframe, output_file):
    # write selfies representation in dataframe to text file
    with open(output_file, 'w') as f:
        for selfie in dataframe['selfies']:
            f.write("%s\n" % selfie)


def smiles_fragments_to_selfies_fragments(smiles_fragments_list):
    # convert smi_fragments column to SELFIES
    selfies_fragments = []
    # convert each element in list of list to SELFIES
    for mol in smiles_fragments_list:
        fragmented_mols_selfies = []
        # split mol with comma as delimiter
        frags = mol.split(',')
        for frag in frags:
            frag = sf.encoder(frag)
            fragmented_mols_selfies.append(frag)
            # join SELFIES fragments with comma as delimiter into string representation
        fragmented_mols_selfies = ','.join(fragmented_mols_selfies)
        
        selfies_fragments.append(fragmented_mols_selfies)

    return selfies_fragments

def weighted_compute_metrics(predictions, class_weights):
    pred, labels = predictions
    # convert preds to probabilities
    probs = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)

    preds = np.argmax(pred, axis=1)
    preds = np.where(preds > 0, 1, 0)  # Convert predictions to 1 or 0
    labels = np.where(labels > 0, 1, 0)  # Convert labels to 1 or 0
    # calculate sample weights for the metrics using the weights passed to the function
    sample_weights = np.where(labels > 0, class_weights[1], class_weights[0])
    accuracy = accuracy_score(y_true=labels, y_pred=preds, sample_weight=sample_weights)
    recall = recall_score(y_true=labels, y_pred=preds, average='weighted', sample_weight=sample_weights)
    precision = precision_score(y_true=labels, y_pred=preds, average='weighted', sample_weight=sample_weights)
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted', sample_weight=sample_weights)
    balanced_accuracy = balanced_accuracy_score(y_true=labels, y_pred=preds, sample_weight=sample_weights)
    auc_roc = roc_auc_score(y_true=labels, y_score=preds, average='weighted', sample_weight=sample_weights)

    
    # Calculate cross-entropy loss from probabilities and true labels
    loss = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    # Standard error of the mean (SEM) for the loss
    sem_loss = sem(np.log(probs[np.arange(len(labels)), labels]))
    # Compute the confidence interval using the t-distribution
    confidence_interval = t.interval(0.95, len(labels) - 1, loc=loss, scale=sem_loss)
    # Perplexity is the exponential of the cross-entropy loss
    perplexity = np.exp(loss)
    perplexity_confidence_interval = tuple(np.exp(b) for b in confidence_interval)
    smoothed_perplexity = np.exp(loss/len(labels))
    smoothed_perplexity_confidence_interval = tuple(np.exp(b/len(labels)) for b in confidence_interval)
    
    
    

    # ground_truth: np.array of 0 and 1
    # predictions: np.array of floats of the probability of being class 1
    ground_truth = labels
    predictions = probs[:,1]
    AUC_ROC_delong, (AUC_ROC_delong_lb, AUC_ROC_delong_ub) = calc_auc_ci(ground_truth, predictions)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true=labels, probas_pred=preds, sample_weight=sample_weights)
    auc_prc = auc(recall_curve, precision_curve)
    # Cohen's kappa: a statistic that measures inter-annotator agreement.
    kappa = cohen_kappa_score(y1=labels, y2=preds, sample_weight=sample_weights)

    return {
        "accuracy": accuracy,
        "balanced_accuracy_score": balanced_accuracy,
        "AUC_ROC": auc_roc,
        "AUC_ROC_delong": AUC_ROC_delong,
        "AUC_ROC_ci_delong": (AUC_ROC_delong_lb, AUC_ROC_delong_ub),
        "kappa": kappa,
        "AUC_PRC": auc_prc, # AUC-PRC is appropriate to measure the performance of a classification model for highly imbalanced data
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "loss": loss,
        "perplexity": perplexity,
        "perplexity_confidence_interval": perplexity_confidence_interval,
        "smoothed_perplexity": smoothed_perplexity,
        "smoothed_perplexity_confidence_interval": smoothed_perplexity_confidence_interval,

    }


def compute_metrics(predictions):
    preds, labels = predictions
    # convert preds to probabilities
    probs  = np.exp(preds) / np.exp(preds).sum(-1, keepdims=True)
    preds = np.argmax(preds, axis=1)
    preds = np.where(preds > 0, 1, 0)  # Convert predictions to 1 or 0
    labels = np.where(labels > 0, 1, 0)  # Convert labels to 1 or 0
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    precision = precision_score(y_true=labels, y_pred=preds, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    balanced_accuracy = balanced_accuracy_score(y_true=labels, y_pred=preds)
    auc_roc= roc_auc_score(y_true=labels, y_score=preds, average='weighted')
    precision_curve, recall_curve, _ = precision_recall_curve(y_true=labels, probas_pred=preds)
    auc_prc = auc(recall_curve, precision_curve)
    
    ground_truth = labels
    predictions = probs[:,1]
    AUC_ROC_delong, (AUC_ROC_delong_lb, AUC_ROC_delong_ub) = calc_auc_ci(ground_truth, predictions)

    # Cohen's kappa: a statistic that measures inter-annotator agreement.
    kappa = cohen_kappa_score(y1=labels, y2=preds)

   
    return {
        "accuracy": accuracy,
        "balanced_accuracy_score": balanced_accuracy,
        "AUC_ROC": auc_roc,
        "AUC_ROC_delong": AUC_ROC_delong,
        "AUC_ROC_ci_delong": (AUC_ROC_delong_lb, AUC_ROC_delong_ub),
        "kappa": kappa,
        "AUC_PRC": auc_prc, # AUC-PRC is appropriate to measure the performance of a classification model for highly imbalanced data
        "f1": f1,
        "precision": precision,
        "recall": recall,    
    }


def plot_training_progress_metrics(trainer, model_name, plot_path="./plots/classification/"):
    """
    Plots the training progress (loss vs. epochs) for a given model.
    
    Args:
    - trainer: A Trainer object that has a state with log history.
    - model_name: A string representing the name of the model.
    - plot_path: The path to where the plot should be saved.
    
    Returns:
    - A plt.figure object representing the plot.
    """
    training_logs = trainer.state.log_history
    
    # Separate out training metrics and evaluation metrics
    train_metrics = [log for log in training_logs if 'loss' in log]
    eval_metrics = [log for log in training_logs if 'eval_loss' in log]
    
    train_epochs = [metric['epoch'] for metric in train_metrics]
    train_loss = [metric['loss'] for metric in train_metrics]
    
    eval_epochs = [metric['epoch'] for metric in eval_metrics]
    eval_loss = [metric['eval_loss'] for metric in eval_metrics]

    # Use style for better aesthetics
    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(figsize=(12, 6))
        # Training loss
        ax.plot(train_epochs, train_loss, label="Training Loss", color='royalblue', linewidth=2, alpha=0.7)
        # Evaluation loss
        ax.plot(eval_epochs, eval_loss, label="Evaluation Loss", color='tomato', linewidth=2, alpha=0.7)
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_title(f"Loss vs Epochs for {model_name}", fontsize=16)
        ax.legend(fontsize=12, shadow=True)
        # Hide the top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Add minor gridlines for better visualization
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth=0.5)
        ax.grid(which='minor', linestyle=':', linewidth=0.5)
        
        # Check if directory exists, if not create it
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        print("Saving loss vs epochs plot......")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f"{model_name}_loss_vs_epochs.png"))

    return fig

def plot_combined_evaluation_metrics(trainer, model_name, plot_path="./plots/classification/"):
    """
    Plots the evaluation metrics over epochs for a given model in a single combined plot.
    
    Args:
    - trainer: A Trainer object that has a state with log history.
    - model_name: A string representing the name of the model.
    - plot_path: The path to where the plot should be saved.
    
    Returns:
    - A plt.figure object representing the plot.
    """
    training_logs = trainer.state.log_history
    eval_metrics = [log for log in training_logs if 'eval_loss' in log]
    eval_epochs = [metric['epoch'] for metric in eval_metrics]

    metrics_values_dic ={}

    metrics_to_plot = ['eval_AUC_ROC','eval_AUC_PRC', 'eval_f1', 'eval_balanced_accuracy_score', 'eval_kappa', 'eval_accuracy', 'eval_AUC_ROC_delong','eval_loss']
    colors = ['tomato', 'royalblue', 'forestgreen', 'darkorange', 'purple', 'deeppink', 'darkcyan', 'darkmagenta']
    # Use style for better aesthetics
    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(figsize=(15, 8))
        for idx, metric_name in enumerate(metrics_to_plot):
            metric_values = [metric[metric_name] for metric in eval_metrics]
            # save metric values in a dictionary
            metrics_values_dic[metric_name] = metric_values
            ax.plot(eval_epochs, metric_values, label=metric_name.split('_')[-1], color=colors[idx], linewidth=2, alpha=0.7)

        ax.set_title(f"Evaluation Metrics vs Epochs for {model_name}", fontsize=16)
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Metrics Value", fontsize=14)
        ax.legend(fontsize=12, shadow=True, loc="upper right")
        # Hide the top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Add minor gridlines for better visualization
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth=0.5)
        ax.grid(which='minor', linestyle=':', linewidth=0.5)
        # Check if directory exists, if not create it
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        print("Saving combined evaluation metrics plot......")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f"{model_name}_combined_eval_metrics_vs_epochs.png"))

    return metrics_values_dic

def plot_final_evaluation_metrics_points(trainer, model_name, plot_path="./plots/classification/"):
    # extract metrics from trainer
    validation_results = trainer.evaluate()
    metrics = validation_results
    accuracy = metrics['eval_accuracy']
    balanced_accuracy = metrics['eval_balanced_accuracy_score']
    f1_score = metrics['eval_f1']
    auc_roc = metrics['eval_AUC_ROC']
    auc_roc_delong = metrics['eval_AUC_ROC_delong']
    kappa = metrics['eval_kappa']
    auc_prc = metrics['eval_AUC_PRC']
    precision = metrics['eval_precision']
    recall = metrics['eval_recall']
    # metrics names and their values
    metric_names = ["accuracy", "balanced_accuracy_score", "AUC_ROC","kappa","AUC_PRC", "f1", "precision", "recall", "AUC_ROC_delong"]
    values = [accuracy, balanced_accuracy, auc_roc, kappa, auc_prc, f1_score, precision, recall, auc_roc_delong]
    # plotting
    plt.figure(figsize=(15, 7))
    # Plotting the line connecting the metrics
    plt.plot(metric_names, values, marker='o', linestyle='-', color='deepskyblue', lw=2)
    # Highlighting each metric point using scatter
    for i, v in enumerate(values):
        plt.scatter(metric_names[i], v, color='tomato', s=150) # s sets the size of the scatter point
        plt.text(metric_names[i], v + 0.01, str(round(v, 2)), ha='center', va='bottom', fontweight='bold', color='tomato', fontsize=10)
    # setting x, y labels and title
    plt.xlabel("Metrics", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.title(f"Metrics for {model_name}", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Enhancements
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim(0, 1.2) # You can adjust this if needed to make sure all annotations fit
    # saving the plot
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    print("saving metrics plot......")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"{model_name}_metrics_points.png"))

def plot_final_evaluation_metrics_bars(trainer, model_name, plot_path="./plots/classification/"):
    # 1. Extract metrics from the trainer
    validation_results = trainer.evaluate()
    
    accuracy = validation_results['eval_accuracy']
    balanced_accuracy = validation_results['eval_balanced_accuracy_score']
    f1_score = validation_results['eval_f1']
    auc_roc = validation_results['eval_AUC_ROC']
    kappa = validation_results['eval_kappa']
    auc_prc = validation_results['eval_AUC_PRC']
    precision = validation_results['eval_precision']
    recall = validation_results['eval_recall']

    # 2. Set up metrics for plotting
    metrics = ["accuracy", "balanced_accuracy", "AUC_ROC", "kappa", "AUC_PRC", "f1", "precision", "recall"]
    values = [accuracy, balanced_accuracy, auc_roc, kappa, auc_prc, f1_score, precision, recall]
    
    # 3. Plot metrics
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, color='cornflowerblue', width=0.6)

    # Customize y axis limit
    plt.ylim(0, max(values) + 0.1)

    # Enhance labels and title
    plt.xlabel("Metrics", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.title(f"Metrics for {model_name}", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Gridlines and cleanup
    plt.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # Add annotations on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2),
                 ha='center', va='bottom', color='black', fontsize=10)
    
    # 4. Save the plot
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    print("Saving metrics plot...")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"{model_name}_metrics_bars.png"))




# add decoded tokens_id to the dataset
def add_decoded_tokens(batch, tokenizer):
    batch["decoded_tokens"] = tokenizer.decode(batch["input_ids"])
    return batch

def save_model(model, model_path_name):
    try:
        with open(model_path_name, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path_name}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(model_path_name):
    try:
        with open(model_path_name, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None



def preprocess_and_save_scaffold_splits():
    logger.info('Processing and saving scaffold splits for training...')

    def process_and_save_data(dataset_name, tasks_list):
        data_processor = Scaffoldprocessor(tasks_wanted=tasks_list, split='scaffold')
        tasks, train_df, valid_df, test_df, transformers = data_processor.process_data(dataset_name)
        
        # Remove rows with NaN values
        train_df = train_df.dropna()
        valid_df = valid_df.dropna()
        test_df = test_df.dropna()

        # Save smiles to text files for training
        smiles_to_text(dataframe=train_df, output_file=f'datasets/pre_processed/{dataset_name}_train.smi')
        smiles_to_text(dataframe=valid_df, output_file=f'datasets/pre_processed/{dataset_name}_val.smi')
        smiles_to_text(dataframe=test_df, output_file=f'datasets/pre_processed/{dataset_name}_test.smi')

    # Process and save ClinTox dataset
    clintox_tasks = ['FDA_APPROVED', 'CT_TOX']
    process_and_save_data("clintox", clintox_tasks)

    # Process and save Tox21 dataset
    tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 
                   'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    process_and_save_data("tox21", tox21_tasks)




# split Data Frame into train,val,test 
def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.8, frac_val=0.1, frac_test=0.1,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''
    
    df_input = df_input.copy()
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))
    
    
    

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)
    
    # # start from zero index
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)   
    df_test = df_test.reset_index(drop=True)

    return df_train, df_val, df_test    




#####################################           AUGMENTATION           #########################################

# SELFIES Augmentation
def enumerate_selfies(selfies_string):
    """
    Enumerates SELFIES for a given SELFIES string.
    """
    smiles = sf.decoder(selfies_string)
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [selfies_string]

    isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(mol))
    isomer_smiles = [Chem.MolToSmiles(isomer) for isomer in isomers]
    isomer_selfies = [sf.encoder(s) for s in isomer_smiles]
    return isomer_selfies

def generate_non_canonical_selfies(selfies_string, num_variants=1):
    """
    Generates non-canonical SELFIES from a given canonical SELFIES.
    """
    smiles = sf.decoder(selfies_string)
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [selfies_string] * num_variants
    
    variants_smiles = [Chem.MolToSmiles(mol, canonical=False, doRandom=True) for _ in range(num_variants)]
    variants_selfies = [sf.encoder(variant) for variant in variants_smiles]
    return variants_selfies


def augment_selfies(tr_text, tr_labels, augmentation_factor):
    
    def perform_augmentation(selfies_list, num_augmentations, label_to_append):
        unique_augmented_selfies = set()
        for _ in range(num_augmentations):
            selected_selfies = random.choice(selfies_list)
            enumerated = enumerate_selfies(selected_selfies)
            for enum_selfies in enumerated:
                non_canonicals = generate_non_canonical_selfies(enum_selfies, augmentation_factor)
                unique_augmented_selfies.update(non_canonicals)
                
        augmented_selfies = list(unique_augmented_selfies)
        tr_text.extend(augmented_selfies)
        tr_labels.extend([label_to_append] * len(augmented_selfies))
        return tr_text, tr_labels

    toxic_selfies = [s for s, l in zip(tr_text, tr_labels) if l == 1]
    non_toxic_selfies = [s for s, l in zip(tr_text, tr_labels) if l == 0]
    
    num_toxic = len(toxic_selfies)
    num_non_toxic = len(non_toxic_selfies)
    logger.info(f"Number of toxic SELFIES: {num_toxic}")
    logger.info(f"Number of non-toxic SELFIES: {num_non_toxic}")

    if num_non_toxic > num_toxic:
        augmentations_needed = (num_non_toxic - num_toxic * augmentation_factor) // augmentation_factor
        logger.info(f"Number of augmentations needed for toxic class: {augmentations_needed}")
        tr_text, tr_labels = perform_augmentation(toxic_selfies, augmentations_needed, 1)
    else:
        augmentations_needed = (num_toxic - num_non_toxic * augmentation_factor) // augmentation_factor
        logger.info(f"Number of augmentations needed for non-toxic class: {augmentations_needed}")
        tr_text, tr_labels = perform_augmentation(non_toxic_selfies, augmentations_needed, 0)

    return tr_text, tr_labels


# SMILES augmentation 
def generate_non_canonical_smiles(smiles, num_variants=1):
    """
    Generates non-canonical SMILES from a given canonical SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [smiles] * num_variants  # Return the original if it can't be parsed
    
    variants = [Chem.MolToSmiles(mol, canonical=False, doRandom=True) for _ in range(num_variants)]
    return variants

# SMILES ENUMERATION 
def enumerate_smiles(smiles):
    """
    Enumerates SMILES for a given SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [smiles]  # Return the original if it can't be parsed

    isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(mol))
    return [Chem.MolToSmiles(isomer) for isomer in isomers]


def augment_smiles(tr_text, tr_labels, augmentation_factor):
    
    def perform_augmentation(smiles_list, num_augmentations, label_to_append):
        unique_augmented_smiles = set()
        for _ in range(num_augmentations):
            selected_smiles = random.choice(smiles_list)
            enumerated = enumerate_smiles(selected_smiles)
            for enum_smiles in enumerated:
                non_canonicals = generate_non_canonical_smiles(enum_smiles, num_variants=augmentation_factor)
                unique_augmented_smiles.update(non_canonicals)
                
        augmented_smiles = list(unique_augmented_smiles)
        tr_text.extend(augmented_smiles)
        tr_labels.extend([label_to_append] * len(augmented_smiles))
        return tr_text, tr_labels

    toxic_smiles = [s for s, l in zip(tr_text, tr_labels) if l == 1]
    non_toxic_smiles = [s for s, l in zip(tr_text, tr_labels) if l == 0]
    
    num_toxic = len(toxic_smiles)
    num_non_toxic = len(non_toxic_smiles)
    logger.info(f"Number of toxic SMILES: {num_toxic}")
    logger.info(f"Number of non-toxic SMILES: {num_non_toxic}")

    if num_non_toxic > num_toxic:
        augmentations_needed = (num_non_toxic - num_toxic * augmentation_factor) // augmentation_factor
        logger.info(f"Number of augmentations needed for toxic class: {augmentations_needed}")
        tr_text, tr_labels = perform_augmentation(toxic_smiles, augmentations_needed, 1)
    else:
        augmentations_needed = (num_toxic - num_non_toxic * augmentation_factor) // augmentation_factor
        logger.info(f"Number of augmentations needed for non-toxic class: {augmentations_needed}")
        tr_text, tr_labels = perform_augmentation(non_toxic_smiles, augmentations_needed, 0)

    return tr_text, tr_labels





######################## PRE-TRAINING AUGMENTATION #############################

def augment_smiles_without_labels(tr_text, augmentation_factor):
    augmented_smiles_list = []

    for smiles in tr_text:
        # Enumerate SMILES before generating non-canonical versions
        enumerated = enumerate_smiles(smiles)
        
        for enum_smiles in enumerated:
            non_canonicals = generate_non_canonical_smiles(enum_smiles, num_variants=augmentation_factor)
            augmented_smiles_list.extend(non_canonicals)

    # Combine the original and augmented data
    tr_text.extend(augmented_smiles_list)

    return tr_text


def augment_selfies_without_labels(tr_text, augmentation_factor):
    augmented_selfies_list = []

    for selfies in tr_text:
        # Enumerate SELFIES before generating non-canonical versions
        enumerated = enumerate_selfies(selfies)

        for enum_selfies in enumerated:
            non_canonicals = generate_non_canonical_selfies(enum_selfies, augmentation_factor)
            augmented_selfies_list.extend(non_canonicals)

    # Combine the original and augmented data
    tr_text.extend(augmented_selfies_list)

    return tr_text
