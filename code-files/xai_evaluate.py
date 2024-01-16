import logging
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np
from copy import deepcopy
from scipy.spatial.distance import pdist, cdist
from xai_approaches import attention_scores_approach, LIME_approach, integrated_gradients_approach, shapley_approach
import logging  
logger = logging.getLogger(__name__)

# EVALUATION FUNCTIONS

def process_get_explanations_duplicates(explanations, tokenizer, sample):
   # FIX DUPLICATES
    sample_tokens = tokenizer.tokenize(sample)
    # extract duplicates from the sample tokens
    duplicates = set([token for token in sample_tokens if sample_tokens.count(token) > 1])
    # make sure the duplicates are duplicated in the explanations as well
    for method, (values, labels) in explanations.items():
        # remove the special tokens from the fragments
        labels = [re.sub(r'^\s+', '', label) for label in labels]
        # convert the values to list if not list
        if not isinstance(values, list):
            values = values.tolist()
        if not isinstance(labels, list):
            labels = labels.tolist()
        # check if the duplicates are duplicated in the explanations as well
        for duplicate in duplicates:
            if duplicate in labels:
                # get the index of the duplicate
                index = labels.index(duplicate)
                # duplicate the value and label
                values.insert(index, values[index])
                labels.insert(index, labels[index])
            else:
                # If the duplicate is not in the labels, add it to the labels and add a zero value
                labels.append(duplicate)
                values.append(0)

        explanations[method] = (values, labels)

    
    return explanations
    
def sort_explanations_by_value(explanations , tokenizer, sample):
     
   
    # SORT EXPLANATIONS
    sorted_explanations = {}
    for method, (values, labels) in explanations.items():
        sorted_indices = np.argsort(values)[::-1]  # Sort in descending order
        
        sorted_values = [values[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]

        # remove the special tokens from the fragments
        sorted_labels = [re.sub(r'^\s+', '', label) for label in sorted_labels]
        # convert the values to list if not list
        if not isinstance(sorted_values, list):
            sorted_values = sorted_values.tolist()
        if not isinstance(sorted_labels, list):
            sorted_labels = sorted_labels.tolist()

        
        sorted_explanations[method] = (sorted_values, sorted_labels)
    
    return sorted_explanations

def get_explanations(method_name, sample, tokenizer, model, target_class=1):
    """
    Get explanations for a given sample using different XAI approaches.
    
    Parameters:
    - method_name: The name of the XAI method to use.
    - sample: The sample for which explanations will be generated.
    - tokenizer: The tokenizer used to process the input data.
    - model: The model for which explanations will be generated.
    
    Returns:
    - explanations: A dictionary containing the explanations for the given sample.

    """
    # ATTENTION SCORES APPROACH
    if method_name == 'attention_scores':
        attention_scores_max, attention_scores_avg, mol_ids = attention_scores_approach(sample, tokenizer, model)
        # remove the special tokens from the fragments
        attention_scores_max = attention_scores_max.detach().cpu().numpy() if torch.is_tensor(attention_scores_max) else attention_scores_max
        attention_scores_avg = attention_scores_avg.detach().cpu().numpy() if torch.is_tensor(attention_scores_avg) else attention_scores_avg
        # shift the scores centered around 0
        attention_scores_max = attention_scores_max - np.mean(attention_scores_max)
        attention_scores_avg = attention_scores_avg - np.mean(attention_scores_avg)
        attention_labels = tokenizer.convert_ids_to_tokens(mol_ids)

        explanations = {
            'attention_max': (attention_scores_max, attention_labels),
            'attention_avg': (attention_scores_avg, attention_labels)
        }
   
    # LIME APPROACH
    elif method_name == 'LIME':
        exp = LIME_approach(sample, tokenizer, model)
        # Extract values and labels from LIME explanation
        
        lime_values = [x[1] for x in exp.as_list()]
        lime_labels = [x[0] for x in exp.as_list()]
        
        explanations = {
            'LIME': (lime_values, lime_labels)
        }

        
    # INTEGRATED GRADIENTS APPROACH
    elif method_name == 'IG':
        attributions_sum, IG_tokens, _ = integrated_gradients_approach(sample, tokenizer, model, target_class)
        attributions_sum = attributions_sum.detach().cpu().numpy() if torch.is_tensor(attributions_sum) else attributions_sum
        explanations = {
            'IG': (attributions_sum, IG_tokens)
        }
    
    
    # SHAPLEY APPROACH
    elif method_name == 'shapley':
        shaps = shapley_approach(sample, tokenizer, model, target_class=target_class)
        explanations = {
            'shapley': (shaps[0], shaps[1])
        }

    # FIX DUPLICATES
    sample_tokens = tokenizer.tokenize(sample)
    if len(set(sample_tokens)) != len(sample_tokens) and method_name == 'LIME':
       explanations =  process_get_explanations_duplicates(explanations, tokenizer, sample)

    # sort the explanations based on the importance values
    explanations = sort_explanations_by_value(explanations, tokenizer , sample)

    

    return explanations


############################################## EVALUATION METRICS #########################################################


# METHOD 1 : SIMILAR PAIRS CONSISTENCY
def get_similar_mol_pairs(toxic_mols, threshold=0.6):
    """Find pairs of molecules that are similar based on their Tanimoto similarity."""
    mols = [Chem.MolFromSmiles(smiles) for smiles in toxic_mols if Chem.MolFromSmiles(smiles)]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024) for mol in mols]

    similarity_matrix = [[DataStructs.TanimotoSimilarity(fp1, fp2) for fp2 in fingerprints] for fp1 in fingerprints]
    similar_molecule_pairs = [(toxic_mols[i], toxic_mols[j]) for i in range(len(similarity_matrix)) for j in range(i+1, len(similarity_matrix)) if similarity_matrix[i][j] > threshold]
    
    logger.info(f' length of similar molecule pairs : {len(similar_molecule_pairs)}')
    return similar_molecule_pairs

def similar_pairs_consistency(toxic_mols, method_name, method, tokenizer, classifier):
    """
    Calculate the consistency score for an XAI method.
    """
    similar_pairs = get_similar_mol_pairs(toxic_mols)
    concordance_scores = []

    for mol1, mol2 in similar_pairs:
        explanations_mol1 = get_explanations(method_name, mol1, tokenizer, classifier)
        explanations_mol2 = get_explanations(method_name, mol2, tokenizer, classifier)

        # Ensure the method exists in the explanations
        if method not in explanations_mol1 or method not in explanations_mol2:
            continue

        values_mol1, values_mol2 = explanations_mol1[method][0], explanations_mol2[method][0]

        # Pad the values to make them of the same length
        max_len = max(len(values_mol1), len(values_mol2))
        values_mol1 = np.pad(values_mol1, (0, max_len - len(values_mol1)))
        values_mol2 = np.pad(values_mol2, (0, max_len - len(values_mol2)))

        # Check the length of the arrays before computing the pearson correlation
        if len(values_mol1) < 2 or len(values_mol2) < 2:
            continue

        similarity, _ = pearsonr(values_mol1, values_mol2)
        concordance_scores.append(similarity)

    # Average the concordance scores
    avg_score = np.mean(concordance_scores) if concordance_scores else 0
    # median score
    median_score = np.median(concordance_scores) if concordance_scores else 0

    return concordance_scores, avg_score, median_score



# METHOD 2 : FIDELITY
# METHOD2.1 : FIDELITY BASIC
def basic_fidelity(classifier, tokenizer, method_name , method, test_samples):
 

    total_fidelity_scores = []
    total_num_features = 0
    
    for sample in test_samples:
        # Tokenize the sample and get the model's output
        inputs = tokenizer(sample, return_tensors="pt", padding=True)
        original_logits = classifier(**inputs).logits
        
        # Get the explanation for the sample using the XAI method
        # explanation = method.get_explanation(sample)
        explanations = get_explanations(method_name, sample, tokenizer, classifier)
        logger.info(f' method name : {method_name}')
        logger.info(f' explanations : {explanations}')
        logger.info(f' method : {method}')
        logger.info(f' explanations[method] : {explanations[method]}')

        # For each feature in the explanation, perturb the feature and get the model's output
        importance_array, feature_list = explanations[method]
        for importance, feature in zip(importance_array, feature_list):
            # perturbed_sample, replace_with = perturb_sample(sample, feature, tokenizer, method)
            # perturbed_sample, _ = perturb_sample(perturbed_sample,replace_with, tokenizer, method)
            perturbed_sample = perturb_input(sample,tokenizer,method,perturbation_factor=0.1)
            perturbed_inputs = tokenizer(perturbed_sample, return_tensors="pt", padding=True)
            perturbed_logits = classifier(**perturbed_inputs).logits
            # Calculate the fidelity score for this feature
            # Here, it's a simple difference between original and perturbed output, weighted by the importance
            # You might need a more sophisticated approach depending on your use case
            logger.info(f' original output : {original_logits}')
            logger.info(f' perturbed output : {perturbed_logits}')
            fidelity_score = (torch.abs(original_logits - perturbed_logits) * abs(importance)).sum()
            total_fidelity_scores.append(fidelity_score.item())
            total_num_features += len(feature_list)

    # Average the total fidelity score by the number of features in the explanations
    num_features = total_num_features 
    average_fidelity_score = np.mean(total_fidelity_scores) 
    
    
    return average_fidelity_score, total_fidelity_scores

# METHOD2.2 : FIDELITY PEARSON CORRELATION
def pearson_correlation_fidelity(classifier, tokenizer, method_name, method, test_samples):
    """
    Calculate Pearson correlation as a fidelity metric for XAI methods.

    Parameters:
    - classifier: The trained model for which explanations are being generated.
    - tokenizer: The tokenizer used to process the input data.
    - method_name: The name of the XAI method being evaluated.
    - method: The XAI method being evaluated.
    - test_samples: A list of test samples for which explanations will be generated.

    Returns:
    - fidelity_score: The Pearson correlation fidelity score.
    """

    correlation_scores = []
    for sample in test_samples:
        # Tokenize the sample and get the model's output
        inputs = tokenizer(sample, return_tensors="pt", padding=True)
        original_output = classifier(**inputs).logits
        

        # Get the explanation for the sample using the XAI method
        explanations = get_explanations(method_name, sample, tokenizer, classifier)

        # For each feature in the explanation, perturb the feature and get the model's output
        importance_array, feature_list = explanations[method]
        perturbed_outputs = []

        for importance, feature in zip(importance_array, feature_list):
            # perturbed_sample, _ = perturb_sample(sample, feature, tokenizer, method)
            perturbed_sample = perturb_input(sample,tokenizer,method,perturbation_factor=0.1)
            perturbed_inputs = tokenizer(perturbed_sample, return_tensors="pt", padding=True)
            perturbed_output = classifier(**perturbed_inputs).logits
            perturbed_outputs.append(perturbed_output)

        # Calculate Pearson correlation between original and perturbed outputs
        for perturbed_output in perturbed_outputs:
            correlation, _ = pearsonr(original_output.flatten().detach().numpy(), perturbed_output.flatten().detach().numpy())
            correlation_scores.append(correlation)

    # Calculate the average Pearson correlation as the fidelity score
    fidelity_score = np.mean(correlation_scores)
    return fidelity_score , correlation_scores



    
def custom_tokenizer(toxic_mol, orign_tokenizer):
    fragments = orign_tokenizer.tokenize(toxic_mol)
    # use regular expression to remove ## at the end or the beginning of the fragments
    fragments = [re.sub(r'^##', '', fragment) for fragment in fragments]

    return fragments

def perturb_sample(sample, feature, tokenizer, method):
    """
    Perturbs the given sample by modifying the specified feature.
    
    Parameters:
    - sample: The original sample.
    - feature: The feature to be perturbed.
    
    Returns:
    - perturbed_sample: The sample after perturbation.
    - replaced_with: The fragment that replaced the original feature.
    """
    
    # if (method == 'LIME' or method == 'shapley') and hasattr(tokenizer, 'tokenizer_type') and tokenizer.tokenizer_type == 'WordPiece':
    #     fragments = custom_tokenizer(sample, tokenizer)
    # else:
    
    fragments = tokenizer.tokenize(sample)
        
    if feature not in fragments:
        logger.info(f' feature not in fragments : {feature}')
        return sample, None  # Return the original sample if the feature is not found
    
    # Create a list of unique fragments excluding the feature
    unique_fragments = list(set(fragments) - {feature})
    if not unique_fragments:
        return sample, None  # Return the original sample if no alternative fragments are available
    
    # Replace the feature with a random fragment from the unique fragments list
    replaced_with1 = np.random.choice(unique_fragments)
    potential_replacements = list(set(unique_fragments) - {replaced_with1})
    replaced_with2 = np.random.choice(potential_replacements) if potential_replacements else None
    perturbed_sample = sample.replace(feature, replaced_with1)

    # Log the original sample, feature, replaced fragment, and perturbed sample
    logger.info(f"Original Sample: {sample}")
    logger.info(f"Fragments: {fragments}")
    logger.info(f"Feature: {feature}")
    logger.info(f"Replaced With (Primary): {replaced_with1}")
    if replaced_with2:
        logger.info(f"Replaced With (Secondary): {replaced_with2}")
    logger.info(f"Perturbed Sample: {perturbed_sample}")

    # Returning perturbed sample and secondary replacement
    return perturbed_sample, replaced_with2


# METHOD 3 : REITERATION STABILITY
def pad_arrays(arr1, arr2):
    max_len = max(len(arr1[0]), len(arr2[0]))
    arr1_padded = [x + [0]*(max_len - len(x)) for x in arr1]
    arr2_padded = [x + [0]*(max_len - len(x)) for x in arr2]
    return arr1_padded, arr2_padded

def reiteration_stability(explanations):
    """
    Compute the Reiteration similarity for a set of explanations.
    
    Parameters:
    - explanations (dict): Dictionary containing explanations from different XAI methods.
    # explanations = {method: {'scores': [], 'labels': []} for method in ['attention_max', 'attention_avg', 'IG', 'shapley', 'LIME']}
    
    Returns:
    - similarity_metrics (dict): Dictionary containing average and standard deviation of Jaccard similarities for each method.
    """
    stability_metrics = {}
    
    # Helper function to binarize explanations
    def binarize_explanation(expl):
        # Ensure expl is iterable
        if not isinstance(expl, (list, np.ndarray)):
            expl = [expl]
        return [1 if val != 0 else 0 for val in expl]
    
    
    for  method, (scores, _) in explanations.items():
        bin_expl = [binarize_explanation(val) for val in scores]
        jaccard_mat = 1 - pdist(np.stack(bin_expl, axis=0), 'jaccard')
        avg_jaccard, std_jaccard = np.mean(jaccard_mat), np.std(jaccard_mat)
        stability_metrics[method] = (avg_jaccard, std_jaccard)
    
    # Comparing LIME and SHAP explanations
    if 'LIME' in explanations and 'shapley' in explanations:
        lime_bin_expl = [binarize_explanation(val) for val in explanations['LIME']['scores']]
        shap_bin_expl = [binarize_explanation(val) for val in  explanations['shapley']['scores']]
        lime_bin_expl, shap_bin_expl = pad_arrays(lime_bin_expl, shap_bin_expl)
        lime_shap_jaccard_mat = 1 - cdist(np.stack(lime_bin_expl, axis=0), np.stack(shap_bin_expl, axis=0), 'jaccard')
        lime_shap_avg_jaccard, lime_shap_std_jaccard = np.mean(lime_shap_jaccard_mat), np.std(lime_shap_jaccard_mat)
        stability_metrics['LIME_SHAP'] = (lime_shap_avg_jaccard, lime_shap_std_jaccard)
    
    # convert the similarity scores into lists
    stability_metrics = {key: list(value) for key, value in stability_metrics.items()}

    
    return stability_metrics

# METHOD 4 : STRUCTURAL ALERTS INTERSECTION 
def structural_alerts_intersection(alerts, samples, method_name, method, tokenizer, model):
    # Convert alerts to a set for faster membership tests
    alerts_set = set(alerts)
    alerted_samples = []
    potential_alerts = set()  # Using a set to avoid duplicates
    for sample in samples:
        explanation = get_explanations(method_name, sample, tokenizer, model)
        values = explanation[method][0]
        fragments = explanation[method][1]
        for value, fragment in zip(values, fragments):
            if value > 0:
                potential_alerts.add(fragment)
                if fragment in alerts_set:
                    logger.info(f' fragment in alerts_set : {fragment}')
                    alerted_samples.append(sample)

    logger.info(f'Length of unique potential alerts: {len(potential_alerts)}')
    logger.info(f'Length of alerted samples: {len(alerted_samples)}')


    count = sum(1 for alert in potential_alerts if alert in alerts_set)

    logger.info(f'Count of alerts in potential alerts: {count}')
    logger.info(f'alerted samples : {alerted_samples}')

            
    score = (count / len(potential_alerts)) 

    return score 

# METHOD 6.2 : EVALUATING statbility of post-hoc explanations (how much the explanations change when we make small perturbations to the input)
def evaluate_stability(model, tokenizer, method_name, method, test_samples , target_class=1):
    stability_scores = []
    # samples should be list of strings (if it is not list)
    if not isinstance(test_samples, list):
        raise ValueError("test_samples should be a list of strings")

    for sample in test_samples:
        

        explanations_original = get_explanations(method_name, sample, tokenizer, model, target_class)
        # select fragment randomly
        feature = np.random.choice(explanations_original[method][1])      
        # perturbed_sample, _ = perturb_sample(sample, feature, tokenizer, method)
        perturbed_sample = perturb_input(sample,tokenizer, model, method_name,method,target_class=target_class)
        explanations_perturbed = get_explanations(method_name, perturbed_sample, tokenizer, model, target_class)
        
        logger.info(f' explanations_original : {explanations_original}')
        logger.info(f' explanations_perturbed : {explanations_perturbed}')
        
        
        # Compute similarity between explanations_original and explanations_perturbed
        # This can be done using Jaccard similarity, cosine similarity, etc.
        # jacard similarity
        similarity =  len(set(explanations_original[method][1]).intersection(set(explanations_perturbed[method][1]))) / len(set(explanations_original[method][1]).union(set(explanations_perturbed[method][1])))
        stability_scores.append(similarity)
    
    logger.info(f'stability_scores : {stability_scores}')
    return np.mean(stability_scores)


####################################### NEW METRICS  #########################################

# METHOD 5 : EVALUATING FAITHFULNESS (or correctnes) of post-hoc explanations , Explanations describe the behaviour of the model


def evaluate_faithfulness(model, tokenizer, method_name, method, test_samples, top_k_fraction=0.4, target_class=1):
    faithful_scores = []
    # put the model in evaluation mode
    model.eval()
    model.zero_grad()
    for sample in test_samples:
        inputs = tokenizer(sample, return_tensors="pt", padding=True)
        original_logits= model(**inputs).logits
        predicted_class = torch.argmax(original_logits, dim=1).item()
        original_probs = torch.softmax(original_logits, dim=1)
        original_prob= original_probs.max(dim=1)[0].item()
        explanations = get_explanations(method_name, sample, tokenizer, model, target_class)
        explnation_length = len(explanations[method][1])
        logger.info(f' explnation_length : {explnation_length}')
        logger.info(f' top_k_fraction : {top_k_fraction}')
        top_k = int((explnation_length * top_k_fraction) if (explnation_length * top_k_fraction) >= 1 else 1)
        logger.info(f' top_k : {top_k}')
        top_explanations = explanations[method][1][:top_k]  # Assuming top 3 explanations
        logger.info(f' explanations : {explanations}')
        logger.info(f' top explanations : {top_explanations}')
        logger.info(f' original token : {tokenizer.tokenize(sample)}')
        
        orig_tokens = tokenizer.tokenize(sample)
        # remove top explanations from the original tokens to get the perturbed sample
        weak_tokens = [token for token in orig_tokens if token not in top_explanations]
        weak_sample = tokenizer.convert_tokens_to_string(weak_tokens)
        weak_sample = re.sub(r'\s+', '', weak_sample)
        perturbed_sample = ' '.join(weak_tokens)
        perturbed_sample = re.sub(r'\s+', '', perturbed_sample)
        
        

        
        weak_logits = model(**tokenizer(weak_sample, return_tensors="pt", padding=True)).logits
        weak_predicted_class = torch.argmax(weak_logits, dim=1).item()
        # get the probability of the perturbed prediction
        weak_probs = torch.softmax(weak_logits, dim=1)   
        weak_prob = weak_probs.max(dim=1)[0].item()    
        
        if weak_predicted_class != predicted_class:
            weak_prob = -weak_prob

        else:
            weak_prob = weak_prob

        probs_change = np.abs(original_prob - weak_prob)
        # tensor.detach().numpy() to convert tensor to numpy array
        original_logits = original_logits.detach().numpy()
        weak_logits = weak_logits.detach().numpy()
        logits_change =  np.abs(original_logits-weak_logits).sum()
        faithful_scores.append(probs_change)
    
        
        logger.info(f' original sample : {sample}')
        logger.info(f' original predicted class : {predicted_class}')
        logger.info(f' original prob : {original_prob}')
        logger.info(f' original logits : {original_logits}')
        logger.info(f' weak sample : {weak_sample}')
        logger.info(f' perturbed sample : {perturbed_sample}')
        logger.info(f' weak predicted class : {weak_predicted_class}')
        logger.info(f' weak prop : {weak_prob}')
        logger.info(f' weak logits : {weak_logits}')
        logger.info(f' logits change : {logits_change}')
        logger.info(f' probs change : {probs_change}')
        
        
    
    return faithful_scores, np.mean(faithful_scores) 

# METHOD 6.1 : EVALUATING statbility of post-hoc explanations (how much the explanations change when we make small perturbations to the input)
def mol_to_embeddings(tokenizer, model, mol):
    model_embeddings = model.get_input_embeddings()
    # Tokenize the input text and convert to token IDs
    input_ids = tokenizer.encode(mol, add_special_tokens=True)

    # Convert list of token IDs to a tensor
    input_ids_tensor = torch.tensor([input_ids])

    # Retrieve the embeddings for the input IDs
    embeddings = model_embeddings(input_ids_tensor)

    # Optionally, you can average the embeddings to get a single vector
    # for the entire sequence
    averaged_embeddings = embeddings.mean(dim=1)

    return averaged_embeddings

def  generate_perturbations(sample, tokenizer, model, method_name, method,target_class, number_of_perturbations=3):
    perturbed_samples = []
    for i in range(number_of_perturbations):
        perturbed_sample = perturb_input(sample,tokenizer, model, method_name,method,target_class)
        perturbed_samples.append(perturbed_sample)
    return perturbed_samples

def perturb_input(sample,tokenizer, model, method_name,method,target_class,perturbation_factor=0.1):
    """
    Add slight perturbations to a sample.
    """
    tokens = tokenizer.tokenize(sample)
    perturbed_length = max(1, int(len(tokens) * (1 - perturbation_factor)))

   

    # Create a list of indices to shuffle
    indices = list(range(len(tokens)))
    np.random.shuffle(indices)
    # Take the first perturbed_length indices to remove
    indices_to_chose = set(indices[:perturbed_length])
    # Remove the tokens at the selected indices
    perturbed_tokens = [token for i, token in enumerate(tokens) if i  in indices_to_chose]
        

    # generate perturbed tokens with preserving the order of the tokens
    # perturbed_tokens = np.random.choice(tokens, perturbed_length, replace=False)


    # choose the top perturbed_length tokens with highest importance values
    # explanations = get_explanations(method_name, sample, tokenizer, model, target_class)
    # perturbed_tokens = explanations[method][1][:perturbed_length]  # Assuming top perturbed_length explanations
    
    perturbed_sample = re.sub(r'\s+', '', ' '.join(perturbed_tokens))
    
    # perturbed_length = int(len(tokens))
    # logger.info(f' perturbed length : {perturbed_length}')
    # logger.info(f' original tokens : {tokens}')
    # logger.info(f' original sample : {sample}')
    # logger.info(f' perturbed tokens  : {perturbed_tokens}')
    # logger.info(f' perturbed sample : {perturbed_sample}')

    
    return  perturbed_sample


def calculate_local_lipschitz_constant(original_samples, model, tokenizer, method_name, method,number_of_perturbations, target_class=1):
    local_lipschitz_constants = []
    for original_sample in original_samples:

        origin_exp= get_explanations(method_name, original_sample, tokenizer, model, target_class)
        original_explanation_vector = origin_exp[method][0]
        original_explanation_vector = np.array(original_explanation_vector)
        origin_samples_embeddings = mol_to_embeddings(tokenizer, model, original_sample)
        origin_samples_embeddings = origin_samples_embeddings.detach().cpu().numpy()
        perturbed_samples = generate_perturbations(original_sample, tokenizer, model, method_name, method,target_class, number_of_perturbations)

        lipschitz_constants = []

        for perturbed_sample in perturbed_samples:
            perturbed_exp = get_explanations(method_name, perturbed_sample, tokenizer, model, target_class)
            perturbed_explanation_vector =  perturbed_exp[method][0]
            perturbed_explanation_vector = np.array(perturbed_explanation_vector)

            perturbed_sample_embeddings = mol_to_embeddings(tokenizer, model, perturbed_sample)
            perturbed_sample_embeddings = perturbed_sample_embeddings.detach().cpu().numpy()
            # Calculate the L2 norm of the difference in explanations
            max_len = max(len(original_explanation_vector), len(perturbed_explanation_vector))
            original_explanation_vector = np.pad(original_explanation_vector, (0, max_len - len(original_explanation_vector)))
            perturbed_explanation_vector = np.pad(perturbed_explanation_vector, (0, max_len - len(perturbed_explanation_vector)))
            explanation_distance = np.linalg.norm(original_explanation_vector - perturbed_explanation_vector)

            # Calculate the L2 norm of the difference in embeddings
            embedding_distance = np.linalg.norm(origin_samples_embeddings - perturbed_sample_embeddings)
            
            logger.info(f'processed between {original_sample} and {perturbed_sample}')
            # Avoid division by zero in case of identical embeddings
            if embedding_distance == 0:
                logger.info(f' embedding_distance == 0 : {embedding_distance}')
                continue
            
            # Calculate the ratio which is the local Lipschitz estimate for this perturbation
            local_lipschitz_estimate = explanation_distance / embedding_distance    
            lipschitz_constants.append(local_lipschitz_estimate)
            logger.info(f' local_lipschitz_estimate : {local_lipschitz_estimate} between {original_sample} and {perturbed_sample}')

        # The local Lipschitz constant is the maximum of the local estimates
        logger.info(f' lipschitz_constants : {lipschitz_constants}')
        local_lipschitz_constant = max(lipschitz_constants , default=0)
        local_lipschitz_constants.append(local_lipschitz_constant)
        
    
    return local_lipschitz_constants , np.mean(local_lipschitz_constants)







# METHOD 7 : EVALUATING FAIRNESS of post-hoc explanations (the accuracy of the explanations for different subgroups (majority vs minority) of the data should be kind of similar)
def evaluate_fairness(model, tokenizer, method_name, method, tox_minority, non_tox_majority, top_k_fraction):
    
    instatnility_scores_non_tox, instability_non_tox = calculate_local_lipschitz_constant(non_tox_majority, model, tokenizer, method_name, method,number_of_perturbations=10, target_class=0)
    instatnility_scores_tox , instability_tox = calculate_local_lipschitz_constant(tox_minority, model, tokenizer, method_name, method,number_of_perturbations=10, target_class=1)
    instability_fairness_scores = [abs(a - b) for a, b in zip(instatnility_scores_non_tox, instatnility_scores_tox)]


    # Calculate faithfulness scores for each subgroup
    faithfulness_scores_non_tox, faithfulness_non_tox = evaluate_faithfulness(model, tokenizer, method_name, method, non_tox_majority, top_k_fraction, target_class=0)
    faithfulness_scores_tox, faithfulness_tox = evaluate_faithfulness(model, tokenizer, method_name, method, tox_minority, top_k_fraction, target_class=1)
    faithfulness_fairness_scores = [abs(a - b) for a, b in zip(faithfulness_scores_non_tox, faithfulness_scores_tox)]


    # Calculate stability scores for each subgroup
    # stability_score_subgroup1 = evaluate_stability(model, tokenizer, method_name, method, tox_minority,target_class=1)
    # stability_score_subgroup2 = evaluate_stability(model, tokenizer, method_name, method, non_tox_majority,target_class=0)

    
    # compare the stability scores to evaluate fairness
    instability_fairness =  np.mean(instability_fairness_scores)

    # Compare the mean faithfulness scores to evaluate fairness
    faithfulness_fairness = np.mean(faithfulness_fairness_scores) 

    
    
    # logging 
    logger.info(f'faithfulness_non_tox : {faithfulness_non_tox}')
    logger.info(f'faithfulness_tox : {faithfulness_tox}')
    
    # Stability
    logger.info(f' instability_non_tox : {instability_non_tox}')
    logger.info(f' instability_tox : {instability_tox}')
    
    # fairness
    logger.info(f' faithfulness_fairness : {faithfulness_fairness}')
    logger.info(f' instability_fairness : {instability_fairness}')
    
    # marginal error fron standard error of the mean for faithfulness scores tox 95% confidence interval
    faithfulness_tox_marginal_error = 1.96 * np.std(faithfulness_scores_tox) / np.sqrt(len(faithfulness_scores_tox)) 
    # marginal error fron standard error of the mean for faithfulness scores non tox
    faithfulness_non_tox_marginal_error = 1.96 * np.std(faithfulness_scores_non_tox) / np.sqrt(len(faithfulness_scores_non_tox)) 

    # marginal error fron standard error of the mean for stability scores tox
    instability_tox_marginal_error = 1.96 * np.std(instatnility_scores_tox) / np.sqrt(len(instatnility_scores_tox))
    # marginal error fron standard error of the mean for stability scores non tox
    instability_non_tox_marginal_error = 1.96 * np.std(instatnility_scores_non_tox) / np.sqrt(len(instatnility_scores_non_tox))

    # marginal error fron standard error of the mean for faithfulness fairness
    faithfulness_fairness_marginal_error = 1.96 * np.std(faithfulness_fairness_scores) / np.sqrt(len(faithfulness_fairness_scores)) 
    # marginal error fron standard error of the mean for stability fairness
    instability_fairness_marginal_error = 1.96 * np.std(instability_fairness_scores) / np.sqrt(len(instability_fairness_scores)) 

    

    
    
    return {

        'toxic_samples_faithfulness': faithfulness_tox,
        'toxic_samples_faithfulness_marginal_error': faithfulness_tox_marginal_error,
        'faithfulness_scores_tox': faithfulness_scores_tox,

        'non_toxic_samples_faithfulness': faithfulness_non_tox,
        'non_toxic_samples_faithfulness_marginal_error': faithfulness_non_tox_marginal_error,
        'faithfulness_scores_non_tox': faithfulness_scores_non_tox,
        

        'toxic_samples_instability': instability_tox,
        'toxic_samples_instability_marginal_error': instability_tox_marginal_error,
        'instatnility_scores_tox': instatnility_scores_tox,

        'non_toxic_samples_instability': instability_non_tox,
        'non_toxic_samples_instability_marginal_error': instability_non_tox_marginal_error,
        'instatnility_scores_non_tox': instatnility_scores_non_tox,
        

        'faithfulness_fairness': faithfulness_fairness,
        'faithfulness_fairness_marginal_error': faithfulness_fairness_marginal_error,
        'faithfulness_fairness_scores': faithfulness_fairness_scores,
        'instability_fairness': instability_fairness,
        'instability_fairness_marginal_error': instability_fairness_marginal_error,
        'instability_fairness_scores': instability_fairness_scores,

    }

