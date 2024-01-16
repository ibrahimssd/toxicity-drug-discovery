from toxic_tokenizer import Toxic_Tokenizer
import torch
import pandas as pd
import torch
from bertviz_repo.bertviz import head_view
import numpy as np
from utils import *
from transformers import AutoModelForSequenceClassification
import argparse
import random
import shap
from rdkit import Chem
import  logging
from xai_plotting import plot_bars, plot_combined_heatmap , plot_XAI_scores , plot_violin , plot_stability , plot_fairness_comparison , report_faithfulness_scores, report_local_lipschitz_scores , analyze_fairness_results,plot_retiration_stability_bars
from xai_approaches import explain_instance
from xai_evaluate import   (get_explanations , evaluate_faithfulness , basic_fidelity , pearson_correlation_fidelity , 
                            similar_pairs_consistency , structural_alerts_intersection , reiteration_stability, evaluate_stability,
                            evaluate_fairness, calculate_local_lipschitz_constant)

from xai_explain import explain
from xai_structural_alerts import generate_structural_alerts
# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASET_PATHS = {
    'clintox': './datasets/scaffold_splits/clintox_{}.csv',
    'tox21': './datasets/scaffold_splits/tox21_{}.csv',
    'HIPS': './datasets/HIPS/hips_{}.csv'
}


def load_dataframes(target_data):
    if target_data not in DATASET_PATHS:
        raise ValueError('Train dataset not found')
    train_df = pd.read_csv(DATASET_PATHS[target_data].format('train'))
    valid_df = pd.read_csv(DATASET_PATHS[target_data].format('val'))
    test_df = pd.read_csv(DATASET_PATHS[target_data].format('test'))
    logger.info(f"columns names of {target_data} target dataset: {train_df.columns}")
    return train_df, valid_df, test_df


def get_tokenizer(args, vocab_size):
    if args.tokenizer_type == 'Atom-wise':
        # load atom wise tokenizer
        vocab_file=f'vocabs/atom_vocab/{args.tokenizer_data}_{vocab_size}.txt'
        atomwise_tokenizer = Toxic_Tokenizer(vocab_file=vocab_file, tokenizer_type='Atom-wise')
        return atomwise_tokenizer , vocab_size
    elif args.tokenizer_type == 'MacFrag':
        # load chemical rules based tokenizer
        vocab_file = f'vocabs/macFrag_vocab/{args.tokenizer_data}_vocab.smi'
        MacFrag_tokenizer = Toxic_Tokenizer(vocab_file=vocab_file, tokenizer_type='MacFrag')
        vocab_size =  MacFrag_tokenizer.vocab_size
        return MacFrag_tokenizer , vocab_size
    elif args.tokenizer_type == 'Morfessor':
        morf_model_pt= f'models/tokenizers/morfessors/morf_{args.tokenizer_data}_{vocab_size}.bin'
        morf_vocab_pt = f'./vocabs/morf_vocab/morf_{args.tokenizer_data}_{vocab_size}.txt'
        morfessor_tokenizer = Toxic_Tokenizer(vocab_file=morf_vocab_pt,
                                                tokenizer_path=morf_model_pt,
                                                tokenizer_type='Morfessor')
        return morfessor_tokenizer , vocab_size
    elif args.tokenizer_type == 'WordPiece':
        # load Data Driven base tokenizers
        wordpiece_model_pt = f"./models/tokenizers/wordpiece/wordpiece_{args.tokenizer_data}_{vocab_size}.bin"
        wordpiece_vocab_pt = f'vocabs/wordpiece_vocab/{args.tokenizer_data}_{vocab_size}.txt'
        wordpiece_tokenizer = Toxic_Tokenizer(vocab_file= wordpiece_vocab_pt,
                                                tokenizer_path=wordpiece_model_pt,
                                               tokenizer_type='WordPiece')
        return wordpiece_tokenizer , vocab_size
    elif args.tokenizer_type == 'BPE':
        # load Data Driven base tokenizers
        Bpe_model_pt = f'models/tokenizers/bpe/{args.tokenizer_data}_{vocab_size}.bin'
        Bpe_vocab_pt = f'./vocabs/bpe_vocab/{args.tokenizer_data}_{vocab_size}.txt'
        Bpe_tokenizer = Toxic_Tokenizer(vocab_file=Bpe_vocab_pt,
                                                tokenizer_path=Bpe_model_pt,
                                                tokenizer_type='BPE')
        return Bpe_tokenizer , vocab_size
    elif args.tokenizer_type == 'SPE':
        # load Data Driven base tokenizers
        spe_file = f'vocabs/spe_vocab/{args.tokenizer_data}_{vocab_size}'
        vocab_file = f'vocabs/spe_vocab/{args.tokenizer_data}_{vocab_size}.txt'
        spe_tokenizer = Toxic_Tokenizer(vocab_file= vocab_file, spe_file= spe_file, tokenizer_type='SPE')
        return spe_tokenizer , vocab_size
    else:
        raise ValueError('Tokenizer not found')

def rank_molecules_by_convertible_fragments(toxic_mols_predicted_toxic, tokenizer):
    molecule_rankings = {}

    for toxic_mol in toxic_mols_predicted_toxic:
        fragments = tokenizer.tokenize(toxic_mol)
        graphable_fragments_count = 0
        total_fragments_count = 0

        for fragment in fragments:
            if fragment != '[UNK]':
                total_fragments_count += 1
                try:
                    mol = Chem.MolFromSmiles(fragment)
                    if mol is not None:  # Successfully converted to a graph
                        graphable_fragments_count += 1
                except:
                    logger.info(f'Fragment: "{fragment}" could not be converted into a graph.')

        # Calculate the percentage of graphable fragments
        if total_fragments_count > 0:
            percentage_graphable = (graphable_fragments_count / total_fragments_count) * 100
        else:
            percentage_graphable = 0

        molecule_rankings[toxic_mol] = percentage_graphable

    # Sort the molecules based on the percentage of graphable fragments
    ranked_molecules = sorted(molecule_rankings.items(), key=lambda item: item[1], reverse=True)

    # append fragments to the ranked molecules
    ranked_molecules = [(mol, percentage_graphable, tokenizer.tokenize(mol)) for mol, percentage_graphable in ranked_molecules]

    return ranked_molecules



def mols_extraction(tokenizer, classifier,mols, toxic_mols, non_toxic_mols):
    # Initialize lists to store results
    predicted_results = {
        "toxic_predicted_toxic": [],
        "toxic_predicted_non_toxic": [],
        "non_toxic_predicted_toxic": [],
        "non_toxic_predicted_non_toxic": [],
        "predicted_toxic": [],
        "predicted_non_toxic": [],
    }

    if (len(toxic_mols)+len(non_toxic_mols)) != len(mols):
        raise ValueError('mols list is not equal to the sum of toxic and non toxic mols lists')

    # Define a helper function for prediction
    def predict_and_classify(mol, is_toxic):
        fragments = tokenizer.tokenize(mol)
        if fragments == ['[UNK]'] or '\\' in mol:
            return
        inputs = tokenizer(mol, return_tensors="pt", padding=True)
        outputs = classifier(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().detach().numpy().round(3)
        predicted_label = 'toxic' if np.argmax(probs) == 1 else 'non_toxic'
        if is_toxic == "None":
            key = f"predicted_{predicted_label}"
        else:
            key = f"{'toxic' if is_toxic else 'non_toxic'}_predicted_{predicted_label}"
        predicted_results[key].append(mol)

    # Process toxic and non-toxic molecules
    for mol in toxic_mols:
        predict_and_classify(mol, is_toxic=True)
    for mol in non_toxic_mols:
        predict_and_classify(mol, is_toxic=False)
    for mol in mols:
        predict_and_classify(mol, is_toxic="None")

    # Logging results
    for key, value in predicted_results.items():
        logger.info(f'Number of {key.replace("_", " ")}: {len(value)}')

    return (predicted_results["toxic_predicted_toxic"], predicted_results["toxic_predicted_non_toxic"],
            predicted_results["non_toxic_predicted_toxic"], predicted_results["non_toxic_predicted_non_toxic"])


# MAIN FUNCTION
def detect_toxicity(args , vocab_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer , vocab_size = get_tokenizer(args, vocab_size)
    logger.info(f'memory summary: {torch.cuda.memory_summary()}')
    logger.info(f'device: {device}')
    logger.info(f'Tokenizer type: {tokenizer.tokenizer_type}')
    logger.info(f'Tokenizer full Vocab size: {tokenizer.vocab_size}')
    logger.info(f'Tokenizer vocab size: {vocab_size}')
    logger.info(f'task: {args.task}')
    logger.info(f'mol representation: {args.mol_rep}')
    logger.info(f'target data: {args.target_data}')
    # load classifier
    check_point  = f'{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{vocab_size}'
    classifier = AutoModelForSequenceClassification.from_pretrained("models/end_points_classifiers/best_" + check_point, num_labels=2)
    classifier.eval()
    classifier.zero_grad()

    # print model summary
    # logger.info(f' model summary : {classifier}')
    
    # Load the dataset
    train_df, valid_df, test_df = load_dataframes(args.target_data)
     
    # Extract toxic mols from the dataset
    tr_mols_toxic = train_df[train_df[f'{args.task}'] == 1][args.mol_rep].tolist()
    val_mols_toxic = valid_df[valid_df[f'{args.task}'] == 1][args.mol_rep].tolist()
    test_mols_toxic = test_df[test_df[f'{args.task}'] == 1][args.mol_rep].tolist()
    logger.info(f' number of toxic training  molecules : {len(tr_mols_toxic)}')
    logger.info(f' number of toxic validation molecules : {len(val_mols_toxic)}')
    logger.info(f' number of toxic testing molecules : {len(test_mols_toxic)}')
    toxic_mols = tr_mols_toxic + val_mols_toxic + test_mols_toxic
    toxic_mols = [mol for mol in toxic_mols if '[UNK]' not in tokenizer.tokenize(mol) and '\\' not in mol]
    logger.info(f' number of filtered toxic molecules : {len(toxic_mols)}')



    # Extract non toxic mols from the dataset
    tr_mols_non_toxic = train_df[train_df[f'{args.task}'] == 0][args.mol_rep].tolist()
    val_mols_non_toxic = valid_df[valid_df[f'{args.task}'] == 0][args.mol_rep].tolist()
    test_mols_non_toxic = test_df[test_df[f'{args.task}'] == 0][args.mol_rep].tolist()
    logger.info(f' number of non toxic training  molecules : {len(tr_mols_non_toxic)}')
    logger.info(f' number of non toxic validation molecules : {len(val_mols_non_toxic)}')
    logger.info(f' number of non toxic testing molecules : {len(test_mols_non_toxic)}')
    non_toxic_mols = tr_mols_non_toxic + val_mols_non_toxic + test_mols_non_toxic
    non_toxic_mols = [mol for mol in non_toxic_mols if '[UNK]' not in tokenizer.tokenize(mol) and '\\' not in mol]
    logger.info(f' number of filtered non toxic molecules : {len(non_toxic_mols)}')

    # collected all mols of training, validation and testing datasets
    mol_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    molecules = mol_df[args.mol_rep].tolist()
    labels = mol_df[args.task].tolist()
    mols = [mol for mol in molecules if '[UNK]' not in tokenizer.tokenize(mol) and '\\' not in mol]
    labels= mol_df[mol_df[args.mol_rep].isin(mols)][args.task].tolist()
    logger.info(f' number of filtered all molecules : {len(mols)}')
    logger.info(f' number of labels : {len(labels)}')
    
    
    # raise error if the sum of toxic and non toxic mols is not equal to the total number of mols
    if (len(toxic_mols)+len(non_toxic_mols)) !=len(mols):
        raise ValueError('the sum of toxic and non toxic mols is not equal to the total number of mols')

    # extract toxic and non toxic mols from the dataset
    toxic_mols_predicted_toxic , toxic_mols_predicted_non_toxic , non_toxic_mols_predicted_toxic , non_toxic_mols_predicted_non_toxic = mols_extraction(tokenizer, classifier, mols, toxic_mols, non_toxic_mols)    
    logger.info(f' number of toxic molecules predicted toxic : {len(toxic_mols_predicted_toxic)}')
    logger.info(f' number of toxic molecules predicted non toxic : {len(toxic_mols_predicted_non_toxic)}')
    logger.info(f' number of non toxic molecules predicted toxic : {len(non_toxic_mols_predicted_toxic)}')
    logger.info(f' number of non toxic molecules predicted non toxic : {len(non_toxic_mols_predicted_non_toxic)}')
    
     

    # # EXPALINABILITY
    # # pick randomly one toxic molecule
    # TOXIC_MOL = 'CC[Sn](Br)(CC)CC' #toxic_mols_predicted_toxic[4] #random.choice(toxic_mols_predicted_toxic)
    # # NONE_TOXIC_MOL = toxic_mols_predicted_non_toxic[3]
    # explainable_molecule = TOXIC_MOL
    # logger.info(f' explainable molecule : {explainable_molecule}')
    # logger.info(f' explainable molecule fragments : {tokenizer.tokenize(explainable_molecule)}')

    # XAI METHODS
    xai_methods = {
                    
                    'LIME': 'LIME',
                    'attention_max': 'attention_scores',
                    'attention_avg': 'attention_scores',
                    'IG': 'IG',
                    'shapley': 'shapley', 
                    }


    # # EXPLAINLE FRAGMENTS PLOTS [METHOD 1]
    # inputs = tokenizer(explainable_molecule, return_tensors="pt", padding=True)
    # outputs = classifier(**inputs)
    # probs = torch.softmax(outputs.logits, dim=-1).cpu().detach().numpy().round(3)
    # predicted_label = 'toxic' if np.argmax(probs) == 1 else 'non_toxic'
    # target_class = 1 if predicted_label == 'toxic' else 0
    # explanations_1 = explain_instance(instance=explainable_molecule, tokenizer=tokenizer, model=classifier, num_reps=1,target_class=target_class)
    
    # logger.info(f' explanations : {explanations_1}')
    # logger.info(f' LIME explanations : {explanations_1["LIME"]}')
    # logger.info(f' IG explanations : {explanations_1["IG"]}')
    # logger.info(f' shapley explanations : {explanations_1["shapley"]}')
    # logger.info(f' attention_max explanations : {explanations_1["attention_max"]}')
    # logger.info(f' attention_avg explanations : {explanations_1["attention_avg"]}')
    # plot_combined_heatmap(explanations_1, file_name=f'mol_heatmap_explanations_{args.tokenizer_type}_{args.task}.png', molecule=explainable_molecule,target_class=target_class)
    # plot_bars(explanations_1['attention_max']['scores'],explanations_1['attention_max']['labels'], "Attention Scores Values", f'mol_attention_max_bars{args.tokenizer_type}_{args.task}.png', explainable_molecule,target_class)
    # plot_bars(explanations_1['attention_avg']['scores'], explanations_1['attention_avg']['labels'], "Attention Scores Values", f'mol_attention_avg_bars{args.tokenizer_type}_{args.task}.png', explainable_molecule,target_class)
    # plot_bars(explanations_1['IG']['scores'], explanations_1['IG']['labels'], "IG Values", f'mol_IG_bars{args.tokenizer_type}_{args.task}.png', explainable_molecule,target_class)
    # plot_bars(explanations_1['LIME']['scores'], explanations_1['LIME']['labels'], "LIME Values", f'mol_LIME_bars{args.tokenizer_type}_{args.task}.png', explainable_molecule,target_class)
    # plot_bars(explanations_1['shapley']['scores'], explanations_1['shapley']['labels'], "Shapley Values", f'mol_shapley_bars{args.tokenizer_type}_{args.task}.png', explainable_molecule,target_class)
    

    
    # # EXPLAINLE FRAGMENTS PLOTS [METHOD 2]
    # Explanations = {method: {'scores': [], 'labels': []} for method in ['attention_max', 'attention_avg', 'IG', 'shapley', 'LIME']}
    # for method, method_name in xai_methods.items():
        
    #     logger.info(f' XAI method : {method_name}')
    #     # get explanations
    #     explanations_2 = get_explanations(method_name, explainable_molecule, tokenizer, classifier,target_class=target_class)
    #     logger.info(f'{method} explanations : {explanations_2[method]}')
    #     plot_bars(explanations_2[method][0], explanations_2[method][1], f'{method} Values', f'mol_{method}_bars{args.tokenizer_type}_{args.task}_sorted.png', explainable_molecule,target_class)

    #     # add explanations to the dictionary
    #     Explanations[method]['scores'].append(explanations_2[method][0]) 
    #     Explanations[method]['labels'].append(explanations_2[method][1])

    # # plot heatmap
    # plot_combined_heatmap(Explanations, file_name=f'mol_heatmap_explanations_{args.tokenizer_type}_{args.task}_sorted.png', molecule=explainable_molecule,target_class=target_class)
    

    ########### ################################# IN PROGRESS ##############################################################


    # ranked_molecules = rank_molecules_by_convertible_fragments(toxic_mols_predicted_toxic, tokenizer)
    # # save the ranked molecules in csv file
    # df = pd.DataFrame(ranked_molecules, columns=['molecule', 'percentage_graphable', 'fragments'])
    # df.to_csv(f'./logs/xai/ranked_molecules_{args.tokenizer_type}_{args.task}.csv', index=False)
    


    logger.info(f' EXPLAINABILITY IN PROGRESS....................')
    path = f'./logs/xai/ranked_molecules_{args.tokenizer_type}_{args.task}.csv'
    # path=f'./logs/xai/fragments_toxic_mols_BPE_NR-AR-LBD_smiles_tox21_10000.csv'
    data = pd.read_csv(path)
    # xai_methods = {'LIME': 'LIME'}
    toxic_mols = data['molecule'].tolist()[0:100]
    for index , explainable_molecule in enumerate(toxic_mols):
        mol_name = f'explaining_molecule_{index+2}'
        explain(args, xai_methods, tokenizer, classifier, explainable_molecule=explainable_molecule, mol_name=mol_name)
    
    
    

    
################################################# EVALUATION ##########################################################
    # # slice mols 
    # # XAI METHODS
    # xai_methods = {
                    
    #                 'LIME': 'LIME',
    #                 'attention_max': 'attention_scores',
    #                 'attention_avg': 'attention_scores',
    #                 'IG': 'IG',
    #                 'shapley': 'shapley', 
    #                 }
    
    # tox_minority = toxic_mols_predicted_toxic
    # non_tox_majority = non_toxic_mols_predicted_non_toxic
    
    # logger.info(f' EVALUATION IN PROGRESS....................')
    
    
    # # [0] REITERATION STABILITY
    # logger.info(f' REITERATION STABILITY IN PROGRESS....................')
    # explanations = explain_instance(instance=TOXIC_MOL, tokenizer=tokenizer, model=classifier, num_reps=10)
    # logger.info(f' explanations : {explanations}')
    # logger.info(f' LIME explanations : {explanations["LIME"]}')
    # statibility = reiteration_stability(explanations)
    # logger.info(f' statibility metric : {statibility}')
    # plot_stability(statibility, f'reteration_stability_{args.tokenizer_type}_{args.task}.png')

    # # SAVE REITERATION STABILITY SCORES
    # df = pd.DataFrame(statibility)
    # df.to_csv(f'./logs/xai/reiteration_stability_{args.tokenizer_type}_{args.task}.csv', index=False)
    


    # # [1] FAITHFULLNESS SCORES (CORRECTNESS)
    # faithfullness_avg = {}
    # faithfullness_distribution = {}
    # logger.info(f' FAITHFULLNESS SCORES IN PROGRESS....................')
    # for method, method_name in xai_methods.items():
    #     faithfullness_scores  , avg_faithfullness_score  = evaluate_faithfulness(classifier, tokenizer, method_name, method, tox_minority, top_k_fraction=0.4)
    #     faithfullness_avg[method] = avg_faithfullness_score
    #     faithfullness_distribution[method] = faithfullness_scores

    #     # SAVE FAITHFULLNESS SCORES
    #     path = f'./logs/xai/faithfullness_scores_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.csv'
        
    #     if not os.path.exists(path):
    #         df = pd.DataFrame(columns=['method', 'faithfullness_avg', 'faithfullness_distribution'])
    #         df = df.append({'method': method, 'faithfullness_avg': avg_faithfullness_score, 'faithfullness_distribution': faithfullness_scores}, ignore_index=True)
    #         df.to_csv(path, index=False)

    #     else:
    #         df = pd.read_csv(path)
    #         df = df.append({'method': method, 'faithfullness_avg': avg_faithfullness_score, 'faithfullness_distribution': faithfullness_scores}, ignore_index=True)
    #         df.to_csv(path, index=False)

    # # sort the faithfullness scores dictionary
    # faithfullness_avg = {k: v for k, v in sorted(faithfullness_avg.items(), key=lambda item: item[1], reverse=True)}
    # faithfullness_distribution = {k: v for k, v in sorted(faithfullness_distribution.items(), key=lambda item: np.median(item[1]), reverse=True)}
    
    # # PLOTTING THE FAITHFULLNESS RESULTS
    # plot_XAI_scores(faithfullness_avg, f'faithfullness_avg_scores_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='faithfullness_avg')
    # plot_violin(faithfullness_distribution, f'faithfullness_distribution_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='faithfullness', legend_location='upper right')

    
    # [2] local_lipschitz_constant POST HOC STABILITY SCORES
    # local_lipschitz_constants = {}
    # local_lipschitz_constant = {}
    # logger.info(f' LOCAL LIPSCHITZ CONSTANT IN PROGRESS....................')
    # for method, method_name in xai_methods.items():
    #     lipschitz_constants, lipschitz_constant = calculate_local_lipschitz_constant(tox_minority, classifier, tokenizer, method_name, method,number_of_perturbations=10, target_class=1)
    #     local_lipschitz_constants[method] = lipschitz_constants
    #     local_lipschitz_constant[method] = lipschitz_constant
    #     logger.info(f' lipschitz constant : {lipschitz_constant}')

    #     # SAVE LOCAL LIPSCHITZ CONSTANT SCORES
    #     path = f'./logs/xai/local_lipschitz_constant_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.csv'
    #     if not os.path.exists(path):
    #         df = pd.DataFrame(columns=['method', 'local_lipschitz_constant', 'local_lipschitz_constant_distribution'])
    #         df = df.append({'method': method, 'local_lipschitz_constant': lipschitz_constant, 'local_lipschitz_constant_distribution': lipschitz_constants}, ignore_index=True)
    #         df.to_csv(path, index=False)
    #     else:
    #         df = pd.read_csv(path)
    #         df = df.append({'method': method, 'local_lipschitz_constant': lipschitz_constant, 'local_lipschitz_constant_distribution': lipschitz_constants}, ignore_index=True)
    #         df.to_csv(path, index=False)

    # # sort the local lipschitz constant scores dictionary
    # local_lipschitz_constants = {k: v for k, v in sorted(local_lipschitz_constants.items(), key=lambda item: np.median(item[1]), reverse=True)}
    # local_lipschitz_constant = {k: v for k, v in sorted(local_lipschitz_constant.items(), key=lambda item: item[1], reverse=True)}

    # # PLOTTING THE LOCAL LIPSCHITZ CONSTANT RESULTS
    # plot_violin(local_lipschitz_constants, f'local_lipschitz_constants_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='local_lipschitz_constants', legend_location='upper right')
    # plot_XAI_scores(local_lipschitz_constant, f'local_lipschitz_constant_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='local_lipschitz_constant')
    
    

    # # [3] FAIRNESS
    # explanations_fairness = {}
    # logger.info(f' FAIRNESS IN PROGRESS....................')
    # for method, method_name in xai_methods.items():
    #     fairness = evaluate_fairness(classifier, tokenizer, method_name, method, tox_minority, non_tox_majority, top_k_fraction=0.4)
    #     explanations_fairness[method] = fairness
    #     logger.info(f' fairness : {fairness}')

    #     # SAVE FAIRNESS SCORES
    #     path = f'./logs/xai/fairness_scores_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.csv'
    #     if not os.path.exists(path):
    #         df = pd.DataFrame(columns=['method', 'fairness'])
    #         df = df.append({'method': method, 'fairness': fairness}, ignore_index=True)
    #         df.to_csv(path, index=False)
    #     else:
    #         df = pd.read_csv(path)
    #         df = df.append({'method': method, 'fairness': fairness}, ignore_index=True)
    #         df.to_csv(path, index=False)

    
    
    # # ################################# IN PROGRESS ########################################
    # # [2] POST HOC STABILITY
    # post_hoc_explainations_stability = {}
    # logger.info(f' POST HOC STABILITY IN PROGRESS....................')
    # for method, method_name in xai_methods.items():
    #     stability_score = evaluate_stability(classifier, tokenizer, method_name, method, tox_minority)
    #     post_hoc_explainations_stability[method] = stability_score
    #     logger.info(f' stability score : {stability_score}')

    #     # SAVE POST HOC STABILITY SCORES
    #     path = f'./logs/xai/post_hoc_explainations_stability_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.csv'
    #     if not os.path.exists(path):
    #         df = pd.DataFrame(columns=['method', 'post_hoc_explainations_stability'])
    #         df = df.append({'method': method, 'post_hoc_explainations_stability': stability_score}, ignore_index=True)
    #         df.to_csv(path, index=False)
    #     else:
    #         df = pd.read_csv(path)
    #         df = df.append({'method': method, 'post_hoc_explainations_stability': stability_score}, ignore_index=True)
    #         df.to_csv(path, index=False)

    # # sort the post hoc stability scores dictionary
    # post_hoc_explainations_stability = {k: v for k, v in sorted(post_hoc_explainations_stability.items(), key=lambda item: item[1], reverse=True)}

    # # PLOTTING THE POST HOC STABILITY RESULTS
    # plot_XAI_scores(post_hoc_explainations_stability, f'post_hoc_explainations_stability_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='post_hoc_explainations_stability')
    


    # # [4] FIDELITY SCORES
    # basic_fidelity_avg = {}
    # basic_fidelity_distribution = {}
    # pearson_correlation_fidelity_avg = {}
    # pearson_correlation_fidelity_distribution = {}
    # logger.info(f' FIDELITY SCORES IN PROGRESS....................')
    # for method, method_name in xai_methods.items():
    #     avg_fidelity_score , xai_fidelity_distribution  = basic_fidelity( classifier, tokenizer, method_name , method, tox_minority)
    #     logger.info(f' fidelity avg : {avg_fidelity_score}')
    #     basic_fidelity_avg[method] = avg_fidelity_score
    #     basic_fidelity_distribution[method] = xai_fidelity_distribution
    #     avg_pearson_correlation_fidelity_score , xai_pearson_correlation_fidelity_distribution  = pearson_correlation_fidelity(classifier, tokenizer, method_name , method, tox_minority)
    #     logger.info(f' pearson correlation fidelity avg : {avg_pearson_correlation_fidelity_score}')
    #     pearson_correlation_fidelity_avg[method] = avg_pearson_correlation_fidelity_score
    #     pearson_correlation_fidelity_distribution[method] = xai_pearson_correlation_fidelity_distribution

    #     # SAVE FIDELITY SCORES
    #     path = f'./logs/xai/fidelity_scores_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.csv'
    #     if not os.path.exists(path):
    #         df = pd.DataFrame(columns=['method', 'basic_fidelity_avg', 'basic_fidelity_distribution', 'pearson_correlation_fidelity_avg', 'pearson_correlation_fidelity_distribution'])
    #         df = df.append({'method': method, 'basic_fidelity_avg': avg_fidelity_score, 'basic_fidelity_distribution': xai_fidelity_distribution, 'pearson_correlation_fidelity_avg': avg_pearson_correlation_fidelity_score, 'pearson_correlation_fidelity_distribution': xai_pearson_correlation_fidelity_distribution}, ignore_index=True)
    #         df.to_csv(path, index=False)
    #     else:
    #         df = pd.read_csv(path)
    #         df = df.append({'method': method, 'basic_fidelity_avg': avg_fidelity_score, 'basic_fidelity_distribution': xai_fidelity_distribution, 'pearson_correlation_fidelity_avg': avg_pearson_correlation_fidelity_score, 'pearson_correlation_fidelity_distribution': xai_pearson_correlation_fidelity_distribution}, ignore_index=True)
    #         df.to_csv(path, index=False)


    # # sort the basic fidelity scores dictionary
    # basic_fidelity_avg = {k: v for k, v in sorted(basic_fidelity_avg.items(), key=lambda item: item[1], reverse=True)}
    # basic_fidelity_distribution = {k: v for k, v in sorted(basic_fidelity_distribution.items(), key=lambda item: np.median(item[1]), reverse=True)}
    # pearson_correlation_fidelity_avg = {k: v for k, v in sorted(pearson_correlation_fidelity_avg.items(), key=lambda item: item[1], reverse=True)}
    # pearson_correlation_fidelity_distribution = {k: v for k, v in sorted(pearson_correlation_fidelity_distribution.items(), key=lambda item: np.median(item[1]), reverse=True)}

    # # PLOTTING THE FIDELITY RESULTS
    # plot_XAI_scores(basic_fidelity_avg, f'basic_fidelity_avg_scores_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='basic_fidelity_avg')
    # plot_violin(basic_fidelity_distribution, f'basic_fidelity_distribution_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='basic_fidelity', legend_location='upper right')
    # plot_XAI_scores(pearson_correlation_fidelity_avg, f'pearson_correlation_fidelity_avg_scores_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='pearson_correlation_fidelity_avg')
    # plot_violin(pearson_correlation_fidelity_distribution, f'pearson_correlation_fidelity_distribution_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='pearson_correlation_fidelity', legend_location='upper right')

    # # [5] SIMILAR PAIRS CONSISTENCY SCORES
    # similar_pairs_consistency_avg = {}
    # similar_pairs_consistency_median = {}
    # similar_pairs_consistency_dist = {}
    # logger.info(f' SIMILAR PAIRS CONSISTENCY SCORES IN PROGRESS....................')
    # for method, method_name in xai_methods.items():
    #     xai_consistency_scores , avg_consistency_score , median_consistency_score  = similar_pairs_consistency(tox_minority, method_name=method_name , method= method , tokenizer= tokenizer,  classifier=classifier)
    #     similar_pairs_consistency_avg[method] = avg_consistency_score
    #     similar_pairs_consistency_median[method] = median_consistency_score
    #     similar_pairs_consistency_dist[method] = xai_consistency_scores

    #     # SAVE SIMILAR PAIRS CONSISTENCY SCORES
    #     path = f'./logs/xai/similar_pairs_consistency_scores_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.csv'
    #     if not os.path.exists(path):
    #         df = pd.DataFrame(columns=['method', 'similar_pairs_consistency_avg', 'similar_pairs_consistency_median', 'similar_pairs_consistency_dist'])
    #         df = df.append({'method': method, 'similar_pairs_consistency_avg': avg_consistency_score, 'similar_pairs_consistency_median': median_consistency_score, 'similar_pairs_consistency_dist': xai_consistency_scores}, ignore_index=True)
    #         df.to_csv(path, index=False)
    #     else:
    #         df = pd.read_csv(path)
    #         df = df.append({'method': method, 'similar_pairs_consistency_avg': avg_consistency_score, 'similar_pairs_consistency_median': median_consistency_score, 'similar_pairs_consistency_dist': xai_consistency_scores}, ignore_index=True)
    #         df.to_csv(path, index=False)


    # # SORT THE SIMILAR PAIRS CONSISTENCY SCORES DICTIONARY
    # similar_pairs_consistency_avg = {k: v for k, v in sorted(similar_pairs_consistency_avg.items(), key=lambda item: item[1], reverse=True)}
    # similar_pairs_consistency_median = {k: v for k, v in sorted(similar_pairs_consistency_median.items(), key=lambda item: item[1], reverse=True)}
    # similar_pairs_consistency_dist = {k: v for k, v in sorted(similar_pairs_consistency_dist.items(), key=lambda item: np.median(item[1]), reverse=True)}

    # # PLOTTING THE CONSISTENCY RESULTS
    # plot_XAI_scores(similar_pairs_consistency_avg, f'similar_pairs_consistency_avg_scores_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='similar_pairs_consistency_avg')
    # plot_XAI_scores(similar_pairs_consistency_median, f'similar_pairs_consistency_median_scores_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='similar_pairs_consistency_median')
    # plot_violin(similar_pairs_consistency_dist, f'similar_pairs_consistency_dist_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='similar_pairs_consistency', legend_location='upper right')

        

    # # [6] STRUCTURAL ALERTS INTERSECTION SCORES
    # # Filter fragments based on criteria
    # min_occurrences = 3
    # min_size = 2
    # max_size = 40
    # ppv_threshold = 0.5
    # alerts = generate_structural_alerts(mols, min_occurrences, min_size, max_size , ppv_threshold , labels, tokenizer, 
    #                                     atom_tokenizer = Toxic_Tokenizer(vocab_file=f'vocabs/atom_vocab/smilesDB_{163}.txt', tokenizer_type='Atom-wise'))
    # logger.info(f' number of structural alerts : {len(alerts)}')

    # # Evaluate structural alerts intersection
    # structrural_intersection_percentage = {}
    # logger.info(f' STRUCTURAL ALERTS INTERSECTION SCORES IN PROGRESS....................')
    # for method, method_name in xai_methods.items():
    #     percentage_intersection = structural_alerts_intersection(alerts, tox_minority, method_name, method, tokenizer, classifier)
    #     structrural_intersection_percentage[method]  = percentage_intersection
    #     logger.info(f' percentage intersection : {percentage_intersection}')

    #     # SAVE STRUCTURAL ALERTS INTERSECTION SCORES
    #     path = f'./logs/xai/structrural_intersection_percentage_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.csv'
    #     if not os.path.exists(path):
    #         df = pd.DataFrame(columns=['method', 'structrural_intersection_percentage'])
    #         df = df.append({'method': method, 'structrural_intersection_percentage': percentage_intersection}, ignore_index=True)
    #         df.to_csv(path, index=False)
    #     else:
    #         df = pd.read_csv(path)
    #         df = df.append({'method': method, 'structrural_intersection_percentage': percentage_intersection}, ignore_index=True)
    #         df.to_csv(path, index=False)

    
    # # sort the structural alerts intersection scores dictionary
    # structrural_intersection_percentage = {k: v for k, v in sorted(structrural_intersection_percentage.items(), key=lambda item: item[1], reverse=True)}

    # # PLOTTING THE STRUCTURAL ALERTS INTERSECTION RESULTS
    # plot_XAI_scores(structrural_intersection_percentage, f'structrural_intersection_percentage_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png', metric='structrural_intersection_percentage')


    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Toxicity detection')
    parser.add_argument('--tokenizer_type', type=str, default='Atom-wise', help='tokenizer type')
    parser.add_argument('--task', type=str, default='NR-AR-LBD', help='task')
    parser.add_argument('--end_points', nargs='+', default=['NR-AR-LBD'], help='end points')
    parser.add_argument('--mol_rep', type=str, default='smiles', help='molecule representation')
    parser.add_argument('--target_data', type=str, default='chembl', help='target data')
    parser.add_argument('--tokenizer_data', type=str, default='chembl', help='tokenizer data')
    parser.add_argument('--vocab_size', type=int, default=10000, help='vocab size')
    parser.add_argument('--vocab_sizes', nargs='+', default=[10000], help='vocab sizes') 
    args = parser.parse_args()
    
    
    
    # detect_toxicity(args, args.vocab_size)

    # FAIRNESS ANALYSIS 
    path = f'./logs/xai/fairness_scores_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.csv'
    plot_fairness_comparison(path, f'fairness_comparison_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.png')
    
    

    # report_faithfulness_scores('./logs/xai/faithfullness_scores_BPE_NR-AR-LBD_smiles_tox21_10000.csv')
    # report_local_lipschitz_scores('./logs/xai/local_lipschitz_constant_BPE_NR-AR-LBD_smiles_tox21_10000.csv')
    # plot_retiration_stability_bars('./logs/xai/reiteration_stability_BPE_NR-AR-LBD.csv')
    # analyze_fairness_results('./logs/xai/fairness_scores_BPE_NR-AR-LBD_smiles_tox21_10000_1.csv')
