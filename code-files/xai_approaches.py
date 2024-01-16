import torch
import logging
logger = logging.getLogger(__name__)
import numpy as np
import shap
import scipy as sp
import re
import torch
from bertviz_repo.bertviz import head_view
import numpy as np
from utils import *
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz 
from lime.lime_text import LimeTextExplainer
import shap
import scipy as sp
import re
import  logging

def process_instance_duplicates(explanations, instance, tokenizer, processed_method):
    # Initialize processed explanations with the same structure as the input
    processed_explanations = {method: {'scores': [], 'labels': []} for method in explanations.keys()}

    # Copy all the explanations except the one that needs to be processed
    for method in explanations.keys():
        if method != processed_method:
            processed_explanations[method]['scores'] = explanations[method]['scores']
            processed_explanations[method]['labels'] = explanations[method]['labels']

    # Tokenize the instance and find duplicates
    sample_tokens = tokenizer.tokenize(instance)
    duplicates = set([token for token in sample_tokens if sample_tokens.count(token) > 1])

    # Process the specified method
    for scores, labels in zip(explanations[processed_method]['scores'], explanations[processed_method]['labels']):
        # Check and handle duplicates in the explanations
        for duplicate in duplicates:
            duplicate_indices = [i for i, label in enumerate(labels) if label == duplicate]
            for index in sorted(duplicate_indices, reverse=True):
                # Duplicate the value and label at each index
                scores.insert(index + 1, scores[index])
                labels.insert(index + 1, labels[index])

        # Update the processed explanations for the specified method
        processed_explanations[processed_method]['scores'].append(scores)
        processed_explanations[processed_method]['labels'].append(labels)

    return processed_explanations

def explain_instance(instance, tokenizer, model, num_reps=1, target_class=1):   


    # Initialize explanations structure
    explanations = {method: {'scores': [], 'labels': []} for method in ['attention_max', 'attention_avg', 'IG', 'shapley', 'LIME']}
   
    for num_rep in range(num_reps): 
   
        # [1]ATTENTION SCORES APPROACHE
        attention_scores_max , attention_scores_avg , mol_ids = attention_scores_approach(instance, tokenizer, model)
        attention_scores_max = attention_scores_max.detach().cpu().numpy()
        attention_scores_avg = attention_scores_avg.detach().cpu().numpy()
        # shift the scores centered around 0
        attention_scores_max = attention_scores_max - np.mean(attention_scores_max)
        attention_scores_avg = attention_scores_avg - np.mean(attention_scores_avg)
        attention_labels = tokenizer.convert_ids_to_tokens(mol_ids)

        # [2] LIME APPROACH
        exp = LIME_approach(instance, tokenizer, model)
        lime_values = [x[1] for x in exp.as_list()]
        lime_labels = [x[0] for x in exp.as_list()]

      
        
        # [3]INTEGRATED GRADIENTS APPROACH
        IG_attributions , IG_tokens , _ = integrated_gradients_approach(instance, tokenizer, model,target_class)
        IG_attributions = IG_attributions.detach().cpu().numpy()
        # SHAPLEY APPROACH
        shaps = shapley_approach(instance, tokenizer, model,target_class=target_class)
        
        
        # convert all scores to list if not already
        attention_scores_max = attention_scores_max.tolist() if not isinstance(attention_scores_max, list) else attention_scores_max
        attention_scores_avg = attention_scores_avg.tolist() if not isinstance(attention_scores_avg, list) else attention_scores_avg
        attention_labels = attention_labels if not isinstance(attention_labels, list) else attention_labels
        IG_attributions = IG_attributions.tolist() if not isinstance(IG_attributions, list) else IG_attributions
        IG_tokens = IG_tokens if not isinstance(IG_tokens, list) else IG_tokens
        shap_vales = shaps[0] if not isinstance(shaps[0], list) else shaps[0]
        shap_labels = shaps[1] if not isinstance(shaps[1], list) else shaps[1]
        lime_values = lime_values if not isinstance(lime_values, list) else lime_values
        lime_labels = lime_labels if not isinstance(lime_labels, list) else lime_labels
        
        # remove the special tokens from the fragments and corresponding values
        attention_labels = [re.sub(r'^\s+', '', label) for label in attention_labels]
        IG_tokens = [re.sub(r'^\s+', '', label) for label in IG_tokens]
        shap_labels = [re.sub(r'^\s+', '', label) for label in shap_labels]
        lime_labels = [re.sub(r'^\s+', '', label) for label in lime_labels]

        
    


        # add the explanations to the dictionary
        explanations['attention_max']['scores'].append(attention_scores_max)
        explanations['attention_max']['labels'].append(attention_labels)
        explanations['attention_avg']['scores'].append(attention_scores_avg)
        explanations['attention_avg']['labels'].append(attention_labels)
        explanations['IG']['scores'].append(IG_attributions)
        explanations['IG']['labels'].append(IG_tokens)
        explanations['shapley']['scores'].append(shap_vales)
        explanations['shapley']['labels'].append(shap_labels)
        explanations['LIME']['scores'].append(lime_values)
        explanations['LIME']['labels'].append(lime_labels)
        

    # PROCESS DUPLICATES
    # check if method is LIME
    instance_tokens = tokenizer.tokenize(instance)
    processed_method = 'LIME'
    if len(instance_tokens) != len(set(instance_tokens))  and processed_method in explanations.keys():
        explanations = process_instance_duplicates(explanations, instance, tokenizer,processed_method)
        
    
    return explanations





# XAI APPROACHES IMPLEMENTATION

##########################################################   ATTENTION SCORES APPROACH   ##########################################################
# APPROACH 1 : ATTENTION SCORES APPROACH
def attention_scores_approach(toxic_mol, tokenizer, model):
    """
    Extract attention scores for the given molecule using a trained classifier.
    
    Parameters:
    - toxic_mol (str): The input molecule string.
    - tokenizer: The tokenizer compatible with the classifier.
    - classifier: The pretrained model.
    
    Returns:
    Tuple[torch.Tensor, torch.Tensor, List[int]]: A tuple containing max attention scores, 
    average attention scores, and token ids without special tokens.
    """
    logger.info('Processing input molecule with attention scores approach...')
    
    mol_ids = tokenizer.encode(toxic_mol, add_special_tokens=True)
    # logger.info(f'Molecule ids: {mol_ids}')
    # logger.info(f'Molecule id to fragments: {tokenizer.convert_ids_to_tokens(mol_ids)}')
    
    inputs_ids = torch.tensor([mol_ids])
    model.eval()  # Set the model to evaluation mode
    model.zero_grad()  # Clear gradients
    out = model(inputs_ids, output_hidden_states=True, output_attentions=True)

    # Extract attention scores
    attentions = out.attentions
    attention_scores_max = compute_max_attention(attentions)
    attention_scores_avg = compute_avg_attention(attentions)

    # Logging top tokens with highest attention scores
    # log_attention_scores(tokenizer.convert_ids_to_tokens(mol_ids), attention_scores_max, 'max')
    # log_attention_scores(tokenizer.convert_ids_to_tokens(mol_ids), attention_scores_avg, 'avg')

    # Remove special tokens
    return attention_scores_max[1:-1], attention_scores_avg[1:-1], mol_ids[1:-1]

def compute_max_attention(attentions):
    """Compute max attention scores from the given attention tensors."""
    return torch.max(attentions[-1], dim=1)[0].max(dim=1)[0].squeeze(0)
    # aggreagte max attention scores from aggregating all the layers and heads
    
    
    

def compute_avg_attention(attentions):
    """Compute average attention scores from the given attention tensors."""
    avg_attention = torch.stack(attentions).mean(dim=0)
    return avg_attention.max(dim=1)[0].max(dim=1)[0].squeeze(0)
    
    

def log_attention_scores(tokens, scores, mode):
    """Log top tokens with their attention scores."""
    attention_dict = dict(zip(tokens, scores.tolist()))
    sorted_attention = {k: v for k, v in sorted(attention_dict.items(), key=lambda item: item[1], reverse=True)}
    
    logger.info(f'Logging the attention scores ({mode})')
    for idx, (token, score) in enumerate(sorted_attention.items()):
        if idx < 10:  # log only the top 10 tokens
            logger.info(f'Token: {token}, Score: {score}')


##########################################################   INTEGRATED GRADIENTS APPROACH   ##################################################################################################################

# APPROACHE 2 : INTEGRATED GRADIENTS APPROACH
def forward_func(inputs: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Forward function for the model.
    """
    outputs = model(inputs, output_hidden_states=True, output_attentions=True)
    logits = outputs.logits[0]

    # reshape [#example x #classes]
    logits = logits.view(-1, logits.shape[-1])
    return logits


def compute_integrated_gradients(model: torch.nn.Module, tokenizer, molecule: str, target_class: int) -> tuple:
    """
    Compute Integrated Gradients for the given data.
    """
    logger.info('INTEGRATED GRADIENTS APPROACH IN PROGRESS...')
    
    input_ids, baseline_input_ids, all_tokens = construct_input_and_baseline(molecule, tokenizer)
    

    # Get model's input embeddings
    model_embeddings= model.get_input_embeddings()
    
    
    
    # Initialize LayerIntegratedGradients
    lig = LayerIntegratedGradients(lambda x: forward_func(x, model), model_embeddings)
    
    # Compute attributions
    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=baseline_input_ids,
                                        n_steps=500,
                                        return_convergence_delta=True,
                                        target=target_class
                                        )
    
    # log delta 
    # logger.info(f'Convergence Delta: {delta}')
    
    # Summarize attributions
    attributions_sum = summarize_attributions(attributions)
    
     # remove the special tokens from the fragments and corresponding values
    all_tokens = all_tokens[1:-1]
    attributions_sum = attributions_sum[1:-1]
    
    
    # Visualize results
    attribute_score_vis = viz.VisualizationDataRecord(
                            word_attributions=attributions_sum,
                            pred_prob=torch.max(model(input_ids)[0]),
                            pred_class=torch.argmax(model(input_ids)[0]),
                            true_class=target_class,
                            attr_class=molecule,
                            attr_score=attributions_sum.sum(),
                            raw_input_ids=all_tokens,
                            convergence_score=delta)
    

  
    # Setup the figure
    # plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k', frameon=True, tight_layout=True)
    # # Visualize the text
    # viz.visualize_text([attribute_score_vis])
    # # Save the plot
    # plt.savefig('./plots/xai/explain/IG_plot.png')
    # # Close the plot to free up memory
    # plt.close()
    
    

   
    return attributions_sum , all_tokens , attribute_score_vis

def construct_input_and_baseline(text: str, tokenizer) -> (torch.Tensor, torch.Tensor, list):
    max_length = 128
    baseline_token_id = tokenizer.pad_token_id 
    sep_token_id = tokenizer.sep_token_id 
    cls_token_id = tokenizer.cls_token_id 
    text_ids = tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=False, return_tensors='pt').squeeze()
    # ensuring that all tensors are 1-dimensional before concatenating them
    if len(text_ids.shape) == 0:
        text_ids = text_ids.unsqueeze(0)
    input_ids = torch.cat([torch.tensor([cls_token_id]), text_ids, torch.tensor([sep_token_id])])
    token_list = tokenizer.convert_ids_to_tokens(input_ids)
    baseline_input_ids = torch.cat([torch.tensor([cls_token_id]), torch.tensor([baseline_token_id] * len(text_ids)), torch.tensor([sep_token_id])])
    result = (input_ids.unsqueeze(0), baseline_input_ids.unsqueeze(0), token_list)
    # remove the special tokens from the fragments and corresponding values 
    return  result

def summarize_attributions(attributions: torch.Tensor) -> torch.Tensor:
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def integrated_gradients_approach(toxic_mol, tokenizer, model, target_class):
    logger.info(f' INTEGRATED GRADIENTS APPROACH IN PROGRESS....................')
    mol_ids = tokenizer.encode(toxic_mol, add_special_tokens=True)
    # inputs_ids = torch.tensor([mol_ids])
    # logger.info(f' Integrated Gradients probs: {torch.softmax(forward_func(inputs_ids, classifier), dim=-1)}') # the function returns logits
    # Ensure the model is in evaluation mode
    
    model.eval()  # Ensure the model is in evaluation mode
    model.zero_grad()  # Clear gradients
    attributions_sum, IG_tokens , viz  = compute_integrated_gradients(model, tokenizer, toxic_mol,target_class)
    
    

    return attributions_sum , IG_tokens , viz



##########################################################   SHAPLEY APPROACH   ##########################################################
# APPROACH 3 : SHAPLEY APPROACH
def predict_shapley(model, tokenizer, mol, max_length,target_class, return_logits=True):
    """
    Predict function for SHAP analysis.

    Args:
        model: The PyTorch model to use for predictions.
        tokenizer: The tokenizer for preprocessing the input.
        mol: The molecule or list of molecules for prediction.
        max_length: The maximum sequence length for the tokenizer.
        return_logits: If True, returns logits; otherwise, returns probabilities.

    Returns:
        The predicted logits or probabilities for the specified class.
    """
    model.eval()
    encoded_input = tokenizer.batch_encode_plus(mol, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids']

    with torch.no_grad():
        outputs = model(input_ids)[0].cpu().numpy()

    if return_logits:
        # logger.info(f' process the logits: {outputs}')
        logits = sp.special.logit((np.exp(outputs).T / np.exp(outputs).sum(-1)).T[:,target_class])
        return  logits
    else:

        probs = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T[:,target_class]
        return  probs



def compute_shapley_values(model, tokenizer, toxic_mol, max_length,target_class):
    """
    Compute SHAP values for the given data.
    
    Args:
    - model (torch.nn.Module): The PyTorch model for which SHAP values are computed.
    - tokenizer: The tokenizer used for encoding the input data.
    - toxic_mol (list): List of molecules for which SHAP values are computed.
    - max_length (int): Maximum length for padding/truncation.
    
    Returns:
    - list: SHAP values.
    """
    masker = shap.maskers.Text(tokenizer)
    masker.mask_token = tokenizer.mask_token
    
    
    explainer = shap.Explainer(lambda x: predict_shapley(model, tokenizer, x, max_length, target_class, return_logits=True),
                               masker,
                            #    output_names=['non-toxic', 'toxic'],
                               seed=42,
                            )

    

    shap_values = explainer([toxic_mol])    
    
    # logger.info(f' SHAP values: {shap_values[0]}')
    
    # waterfall plot
    # plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k', frameon=True, tight_layout=True)
    # shap.plots.waterfall(shap_values[0], show=False, max_display=20)
    # plt.savefig('./plots/xai/explain/SHAP_waterfall_plot.png')
    # plt.close()

    # #  bar plot
    # plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k', frameon=True, tight_layout=True)
    # shap.plots.bar(shap_values[0], show=False, max_display=20)
    # plt.savefig('./plots/xai/explain/SHAP_bar_plot.png')
    # plt.close()

    # # force plot
    # plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k', frameon=True, tight_layout=True)
    # shap.plots.force(shap_values[0], show=False, matplotlib=True)
    # plt.savefig('./plots/xai/explain/SHAP_force_plot.png')
    # plt.close()

    return shap_values



def shapley_approach(toxic_mol, tokenizer, model, max_length=128, target_class=1):
    logger.info('SHAPLEY APPROACH IN PROGRESS...')
    
    
    model.eval()  # Ensure the model is in evaluation mode
    model.zero_grad()  # Clear gradients
    shapley_values = compute_shapley_values(model, tokenizer, toxic_mol, max_length,target_class)
    
    if not hasattr(shapley_values, 'values') or not hasattr(shapley_values, 'data'):
        logger.error('Unexpected format of SHAP values. Make sure the compute_shapley_values function returns correct data.')
        return None
    
    

    values = shapley_values.values[0]
    tokens = shapley_values.data[0]
    
    # remove the special tokens from the fragments and corresponding shapley values
    shaps = (values[1:-1], tokens[1:-1])
    
    return shaps


##############################################################   LIME APPROACH   ##########################################################
# APPROACH 4: LIME
def custom_tokenizer(toxic_mol, orign_tokenizer):
    fragments = orign_tokenizer.tokenize(toxic_mol)
    # use regular expression to remove ## at the end or the beginning of the fragments
    fragments = [re.sub(r'^##', '', fragment) for fragment in fragments]
    return fragments

def predict_LIME(toxic_mol, model , tokenizer):
    inputs = tokenizer(toxic_mol, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    probs = probs.cpu()
    probs = probs.detach().numpy().round(3)
    return probs

def compute_LIME_explanation(toxic_mol, model, tokenizer, class_names):
    

    explainer = LimeTextExplainer(
                                class_names=class_names, 
                                #   split_expression= lambda mol: custom_tokenizer(mol, tokenizer),
                                split_expression= tokenizer.tokenize,
                                    mask_string=tokenizer.mask_token,
                                    
                                  )
    
    exp = explainer.explain_instance(
                                    toxic_mol, 
                                     lambda mol: predict_LIME(mol, model, tokenizer), 
                                     num_features=len(tokenizer.tokenize(toxic_mol)),
                                     num_samples=10000, # number of samples for training the linear model                                        
                                     )
    
    #  visualize the explanation
    # fig = exp.as_pyplot_figure()
    # exp.show_in_notebook(text=True)
    # plt.savefig('./plots/xai/explain/LIME_plot.png')
    # plt.close()

    return exp
    
  

def LIME_approach(toxic_mol, tokenizer, model):
    logger.info(f' LIME APPROACH IN PROGRESS....................')
    model.eval()  # Ensure the model is in evaluation mode
    model.zero_grad()  # Clear gradients

    for param in model.parameters():
        param.requires_grad = False

    # logger.info(f' LIME probs : {predict_LIME(toxic_mol, model, tokenizer)}')
    class_names = ['non-toxic', 'toxic']
    exp = compute_LIME_explanation(toxic_mol, model, tokenizer, class_names)
    return exp






