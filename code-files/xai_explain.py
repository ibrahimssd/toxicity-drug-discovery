
import logging
import numpy as np
import torch
from xai_plotting import plot_bars, plot_combined_heatmap
from xai_approaches import explain_instance
from xai_evaluate import   get_explanations 


#set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def explain(args, xai_methods, tokenizer, classifier, explainable_molecule, mol_name):

    logger.info(f' Explainable molecule : {explainable_molecule}')
    logger.info(f' Explainable molecule fragments : {tokenizer.tokenize(explainable_molecule)}')
    
    inputs = tokenizer(explainable_molecule, return_tensors="pt", padding=True)
    outputs = classifier(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().detach().numpy().round(3)
    predicted_label = 'toxic' if np.argmax(probs) == 1 else 'non_toxic'
    target_class = 1 if predicted_label == 'toxic' else 0

    # EXPLAINLE FRAGMENTS PLOTS [METHOD 1]
    explanations_1 = explain_instance(instance=explainable_molecule, tokenizer=tokenizer, model=classifier, num_reps=1,target_class=target_class)
    logger.info(f' explanations : {explanations_1}')
    logger.info(f' LIME explanations : {explanations_1["LIME"]}')
    logger.info(f' IG explanations : {explanations_1["IG"]}')
    logger.info(f' shapley explanations : {explanations_1["shapley"]}')
    logger.info(f' attention_max explanations : {explanations_1["attention_max"]}')
    logger.info(f' attention_avg explanations : {explanations_1["attention_avg"]}')
    plot_combined_heatmap(explanations_1, file_name=f'mol_heatmap_explanations_{args.tokenizer_type}_{args.task}_{mol_name}.png', molecule=explainable_molecule,target_class=target_class)
    plot_bars(explanations_1['attention_max']['scores'],explanations_1['attention_max']['labels'], "Attention Scores Attributions", f'mol_attention_max_bars{args.tokenizer_type}_{args.task}_{mol_name}.png', explainable_molecule,target_class)
    plot_bars(explanations_1['attention_avg']['scores'], explanations_1['attention_avg']['labels'], "Attention Scores Attributions", f'mol_attention_avg_bars{args.tokenizer_type}_{args.task}_{mol_name}.png', explainable_molecule,target_class)
    plot_bars(explanations_1['IG']['scores'], explanations_1['IG']['labels'], "IG Attributions", f'mol_IG_bars{args.tokenizer_type}_{args.task}_{mol_name}.png', explainable_molecule,target_class)
    plot_bars(explanations_1['LIME']['scores'], explanations_1['LIME']['labels'], "LIME Attributions", f'mol_LIME_bars{args.tokenizer_type}_{args.task}_{mol_name}.png', explainable_molecule,target_class)
    # plot_bars(explanations_1['shapley']['scores'], explanations_1['shapley']['labels'], "Shapley Attributions", f'mol_shapley_bars{args.tokenizer_type}_{args.task}_{mol_name}.png', explainable_molecule,target_class)
    

    
    # EXPLAINLE FRAGMENTS PLOTS [METHOD 2]
    Explanations = {method: {'scores': [], 'labels': []} for method in ['attention_max', 'attention_avg', 'IG', 'shapley', 'LIME']}
    for method, method_name in xai_methods.items():
        
        logger.info(f' XAI method : {method_name}')
        # get explanations
        explanations_2 = get_explanations(method_name, explainable_molecule, tokenizer, classifier,target_class=target_class)
        logger.info(f'{method} explanations : {explanations_2[method]}')
        plot_bars(explanations_2[method][0], explanations_2[method][1], f'{method} Attributions', f'mol_{method}_bars{args.tokenizer_type}_{args.task}_{mol_name}_sorted.png', explainable_molecule,target_class)

        # add explanations to the dictionary
        Explanations[method]['scores'].append(explanations_2[method][0]) 
        Explanations[method]['labels'].append(explanations_2[method][1])

    # plot heatmap
    # plot_combined_heatmap(Explanations, file_name=f'mol_heatmap_explanations_{args.tokenizer_type}_{args.task}_{mol_name}_sorted.png', molecule=explainable_molecule,target_class=target_class)
    
    