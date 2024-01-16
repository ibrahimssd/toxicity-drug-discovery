# from transformers import AutoTokenizer, AutoModelForSequenceClassification , AutoModelForMaskedLM
# import torch
# import numpy as np
# import pandas as pd


# # FINE TUNING
# # Modify tokenizer to add new tokens , and extend embeddings layers
# # initialize the model
# model_name =  "models/pre_trained_models/Best_Clintox_Spe_chemBERTa"
# classification_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# # get the pretrained tokenizer
# pretrained_tokenizer = AutoTokenizer.from_pretrained(model_name)
# pre_trained_vocab = pretrained_tokenizer.get_vocab()
# custom_vocab = tokenizer.get_vocab()
# old_tokens = pretrained_tokenizer.get_vocab().keys()
# current_tokens = tokenizer.get_vocab().keys()
# new_tokens = set(old_tokens) - set(current_tokens)
# new_tokens = set(pre_trained_vocab.keys()) - set(custom_vocab.keys())
# new_tokens = list(new_tokens)
# new_token_ids = [pre_trained_vocab[token] for token in new_tokens]
# # calculated intersected tokens
# intersected_tokens = set(old_tokens).intersection(set(current_tokens)) # 55 tokens
# print("length of intersected tokens", len(intersected_tokens))
# print("length of new_tokens", len(new_tokens))
# print("length of custom tokenizer before ", len(tokenizer))

# # add new tokens with corresponding ids to the tokenizer
# # for token , token_id in zip(new_tokens, new_token_ids):
# #     tokenizer.add_tokens(token)
# #     # add new tokens to the end of the tokenizer

# # add new tokens to the begining of the tokenizer
# # tokenizer.add_tokens(new_tokens)
# print("length of custom tokenizer after ", len(tokenizer))
# classification_model.resize_token_embeddings(len(tokenizer))
# # Update the model configuration
# classification_model.config.vocab_size = len(tokenizer)

# # Set the IDs of the new tokens
# classification_model.config.pad_token_id = tokenizer.pad_token_id
# classification_model.config.eos_token_id = tokenizer.eos_token_id
# classification_model.config.bos_token_id = tokenizer.bos_token_id
# classification_model.config.sep_token_id = tokenizer.sep_token_id
# classification_model.config.cls_token_id = tokenizer.cls_token_id
# classification_model.config.mask_token_id = tokenizer.mask_token_id
# classification_model.config.unk_token_id = tokenizer.unk_token_id

# # re-arrange tokens in tokenizer (special tokens + new tokens + current tokens)
# # new_token_list = list(tokenizer.get_vocab().keys())
# # new_token_list = new_tokens + new_token_list
# # tokenizer.set_vocab(new_token_list)


# # train ByteLevelBPETokenizer on smiles
# ByteLevel_tokenizer= ByteLevelBPETokenizer()
# ByteLevel_tokenizer.train(files=input_file_smi, vocab_size=vocab_size, min_frequency=2000)
# ByteLevel_tokenizer_pt = './models/tokenizers/ByteLevelBPETokenizers/ByteLevelBPETokenizer_' + str(vocab_size) + '.bin'
# # ByteLevel_tokenizer.save_model(ByteLevel_tokenizer_pt)
# save_model(ByteLevel_tokenizer, ByteLevel_tokenizer_pt)


# # plot for attention scores 
#     fig = plt.figure(figsize=(20, 10))
#     plt.bar(range(len(attention_scores)), attention_scores)
#     plt.xticks(range(len(attention_scores)),all_tokens , rotation=90)
#     plt.savefig('./plots/explain/toxic_mol_attention_scores.png')
    
#     # plot for attributions_sum
#     fig = plt.figure(figsize=(20, 10))
#     plt.bar(range(len(attributions_sum)), attributions_sum)
#     plt.xticks(range(len(attributions_sum)),all_tokens , rotation=90)
#     # color positive attributions in green and negative in red
#     for i in range(len(attributions_sum)):
#         if attributions_sum[i] > 0:
#             plt.gca().get_xticklabels()[i].set_color("green")
#             plt.gca().patches[i].set_facecolor("green")
#         else:
#             plt.gca().get_xticklabels()[i].set_color("red")
#             plt.gca().patches[i].set_facecolor("red")
#     plt.savefig('./plots/explain/toxic_mol_attributions_sum.png')

#     # plot for shap_values
#     fig = plt.figure(figsize=(20, 10))
#     plt.bar(range(len(shap_values[0])), shap_values.values[0])
#     plt.xticks(range(len(shap_values[0])),shap_values.data[0] , rotation=90)
#     for i in range(len(shap_values[0])):
#         if shap_values.values[0][i] > 0:
#             plt.gca().get_xticklabels()[i].set_color("green")
#             plt.gca().patches[i].set_facecolor("green")
#         else:
#             plt.gca().get_xticklabels()[i].set_color("red")
#             plt.gca().patches[i].set_facecolor("red")
#     plt.savefig('./plots/explain/toxic_mol_shap_values.png')

#     # plot for exp LIME values (list of tuples with (x,y)) where x is the fragment and y is the weight
#     fig = plt.figure(figsize=(20, 10))
#     plt.bar(range(len(exp.as_list())), [x[1] for x in exp.as_list()])
#     plt.xticks(range(len(exp.as_list())),[x[0] for x in exp.as_list()] , rotation=90)
#     for i in range(len(exp.as_list())):
#         if exp.as_list()[i][1] > 0:
#             plt.gca().get_xticklabels()[i].set_color("green")
#             plt.gca().patches[i].set_facecolor("green")
#         else:
#             plt.gca().get_xticklabels()[i].set_color("red")
#             plt.gca().patches[i].set_facecolor("red")
#     plt.savefig('./plots/explain/toxic_mol_exp_LIME.png')



# logger.info(f' attentions shape : {attentions[-1].shape}')
# # Calculate max attention scores for the input text ids across all heads and layers
# attention_scores = torch.max(attentions[-1], dim=1)[0]
# logger.info(f' attention scores shape : {attention_scores.shape}')
# # Calculate max attention score for each token
# attention_scores = torch.max(attention_scores, dim=1)[0]
# logger.info(f' attention scores shape : {attention_scores.shape}')
# # Remove the batch dimension if present
# attention_scores = attention_scores.squeeze(0)



# ######################## IMportant #########################################



# # prepare downstraem target datasets (clintox and tox21: Scaffold split)
# logger.info('Preparing downstream target datasets...')

# logger.info('moving scaffolds splits into preprocessed datasets files for Macfrag training...')
# data_processor = Scaffoldprocessor(tasks_wanted=['FDA_APPROVED','CT_TOX'],split='scaffold')
# clintox_tasks, clin_train_df, clin_valid_df, clin_test_df, clin_transformers = data_processor.process_data("clintox")

# data_processor = Scaffoldprocessor(tasks_wanted= ['NR-AR', 'NR-AR-LBD', 'NR-AhR','NR-Aromatase', 
#                                                     'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 
#                                                     'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'], split='scaffold')
# tox21_tasks, tox21_train_df, tox21_valid_df, tox21_test_df, tox21_transformers = data_processor.process_data("tox21")
# # remove rows with nan columns
# clin_train_df = clin_train_df.dropna()
# clin_valid_df = clin_valid_df.dropna()
# clin_test_df = clin_test_df.dropna()
# tox21_train_df = tox21_train_df.dropna()
# tox21_valid_df = tox21_valid_df.dropna()
# tox21_test_df = tox21_test_df.dropna()

# # Add mac fragments to the dataframes
# fragmented_mols_smi = read_segmented_mols('./datasets/pre_processed/clintox_train_mac_fragments')
# clin_train_df['smiles_mac_frags'] = fragmented_mols_smi
# fragmented_mols_smi = read_segmented_mols('./datasets/pre_processed/clintox_valid_mac_fragments')
# clin_valid_df['smiles_mac_frags'] = fragmented_mols_smi
# fragmented_mols_smi = read_segmented_mols('./datasets/pre_processed/clintox_test_mac_fragments')
# clin_test_df['smiles_mac_frags'] = fragmented_mols_smi
# fragmented_mols_smi = read_segmented_mols('./datasets/pre_processed/tox21_train_mac_fragments')
# tox21_train_df['smiles_mac_frags'] = fragmented_mols_smi
# fragmented_mols_smi = read_segmented_mols('./datasets/pre_processed/tox21_valid_mac_fragments')
# tox21_valid_df['smiles_mac_frags'] = fragmented_mols_smi
# fragmented_mols_smi = read_segmented_mols('./datasets/pre_processed/tox21_test_mac_fragments')
# tox21_test_df['smiles_mac_frags'] = fragmented_mols_smi
# # convert SMILES in DataFrame to SELFIES
# clin_train_df = clin_train_df[clin_train_df['smiles'].apply(is_supported_smiles)]
# clin_train_df.loc[:, 'selfies'] = clin_train_df.loc[:, 'smiles'].apply(sf.encoder)
# clin_train_df = clin_train_df[['smiles','smiles_mac_frags','selfies','labels','CT_TOX','FDA_APPROVED' ]]

# clin_valid_df = clin_valid_df[clin_valid_df['smiles'].apply(is_supported_smiles)]
# clin_valid_df.loc[:, 'selfies'] = clin_valid_df.loc[:, 'smiles'].apply(sf.encoder)
# clin_valid_df = clin_valid_df[['smiles','smiles_mac_frags','selfies','labels','CT_TOX','FDA_APPROVED' ]]

# clin_test_df = clin_test_df[clin_test_df['smiles'].apply(is_supported_smiles)]
# clin_test_df.loc[:, 'selfies'] = clin_test_df.loc[:, 'smiles'].apply(sf.encoder)
# clin_test_df = clin_test_df[['smiles','smiles_mac_frags','selfies','labels','CT_TOX','FDA_APPROVED' ]]

# tox21_train_df = tox21_train_df[tox21_train_df['smiles'].apply(is_supported_smiles)]
# tox21_train_df.loc[:, 'selfies'] = tox21_train_df.loc[:, 'smiles'].apply(sf.encoder)
# tox21_train_df = tox21_train_df[['smiles','smiles_mac_frags','selfies','labels','NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53' ]]

# tox21_valid_df = tox21_valid_df[tox21_valid_df['smiles'].apply(is_supported_smiles)]
# tox21_valid_df.loc[:, 'selfies'] = tox21_valid_df.loc[:, 'smiles'].apply(sf.encoder)
# tox21_valid_df = tox21_valid_df[['smiles','smiles_mac_frags','selfies','labels','NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53' ]]

# tox21_test_df = tox21_test_df[tox21_test_df['smiles'].apply(is_supported_smiles)]
# tox21_test_df.loc[:, 'selfies'] = tox21_test_df.loc[:, 'smiles'].apply(sf.encoder)
# tox21_test_df = tox21_test_df[['smiles','smiles_mac_frags','selfies','labels','NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53' ]]
# # print shapes of the dataframes
# logger.info('ClinTox train shape: {}'.format(clin_train_df.shape))
# logger.info('ClinTox valid shape: {}'.format(clin_valid_df.shape))
# logger.info('ClinTox test shape: {}'.format(clin_test_df.shape))
# logger.info('Tox21 train shape: {}'.format(tox21_train_df.shape))
# logger.info('Tox21 valid shape: {}'.format(tox21_valid_df.shape))
# logger.info('Tox21 test shape: {}'.format(tox21_test_df.shape))
# # save the dataframes
# clin_train_df.to_csv('./datasets/scaffold_splits/clintox_train.csv',index=False)
# clin_valid_df.to_csv('./datasets/scaffold_splits/clintox_valid.csv',index=False)
# clin_test_df.to_csv('./datasets/scaffold_splits/clintox_test.csv',index=False)
# tox21_train_df.to_csv('./datasets/scaffold_splits/tox21_train.csv',index=False)
# tox21_valid_df.to_csv('./datasets/scaffold_splits/tox21_valid.csv',index=False)
# tox21_test_df.to_csv('./datasets/scaffold_splits/tox21_test.csv',index=False)
# print('Done!')



# # APPROACH 1 : ATTENTION SCORES APPROACH
# def attention_scores_approach(toxic_mol, tokenizer, classifier):
#     # ATTENTION SCORES APPROACH
#     logger.info(f' ATTENTION AVG & MAX APPROACH IN PROGRESS....................')
#     mol_ids = tokenizer.encode(toxic_mol, add_special_tokens=True)
#     logger.info(f' molecule ids  : {mol_ids}')
#     logger.info(f' molecule id to fragments : {tokenizer.convert_ids_to_tokens(mol_ids)}')
#     # get embeddings for the input text ids
#     inputs_ids = torch.tensor([mol_ids])
#     # ensure the model is in evaluation mode
#     classifier.eval()
#     out = classifier(inputs_ids, output_hidden_states=True, output_attentions=True)
#     embeddings = out.hidden_states[-1]
#     logger.info(f' embeddings shape : {embeddings.shape}')
#     # get logits for the input  ids
#     logits = out.logits
#     # print probs for attention scores
#     probs = torch.softmax(logits, dim=-1).cpu().detach().numpy().round(3)
#     logger.info(f' Attention scores probs : {probs}')
#     logger.info(f' logits size : {logits.shape}')
#     # Get attentions for the molecule input from outputs
#     attentions = out.attentions
    
#     # MAX APPROACH
#     logger.info(f' attentions shape : {attentions[-1].shape}')
#     attention_scores_max = torch.max(attentions[-1], dim=1)[0].max(dim=1)[0].squeeze(0)
#     logger.info(f' attention scores shape : {attention_scores_max.shape}')

#     # AVERGE APPROACH 
#     all_attention = torch.stack(attentions)
#     avg_attention = all_attention.mean(dim=0)
#     attention_scores_avg = avg_attention.max(dim=1)[0].max(dim=1)[0].squeeze(0)
#     logger.info(f' attention scores shape : {attention_scores_avg.shape}')
#     # Get the list of tokens
#     fragments = tokenizer.convert_ids_to_tokens(mol_ids)
#     # Build the dictionary with attention scores max and fragments
#     attention_dict_max = dict(zip(fragments, attention_scores_max.tolist()))
#     # Sort the fragments based on attention scores
#     attention_dict_max = {k: v for k, v in sorted(attention_dict_max.items(), key=lambda item: item[1], reverse=True)}
#     # logging the attention scores
#     logger.info(f' logging the attention scores max ')
#     for idx, (token, score) in enumerate(attention_dict_max.items()):
#         if idx < 10:  # log only the top 10 tokens by attention score
#             logger.info(f' token : {token} , score : {score}')
#     # Build the dictionary with attention scores avg and fragments
#     attention_dict_avg = dict(zip(fragments, attention_scores_avg.tolist()))
#     # Sort the fragments based on attention scores
#     attention_dict_avg = {k: v for k, v in sorted(attention_dict_avg.items(), key=lambda item: item[1], reverse=True)}
#     # logging the attention scores
#     logger.info(f' logging the attention scores avg ')
#     for idx, (token, score) in enumerate(attention_dict_avg.items()):
#         if idx < 10:
#             logger.info(f' token : {token} , score : {score}')
    
#     result = (attention_scores_max, attention_scores_avg, mol_ids)   
#     # remove the special tokens from the fragments and corresponding values
#     result = (attention_scores_max[1:-1], attention_scores_avg[1:-1], mol_ids[1:-1])
#     return  result


# # APPROACH 3 : SHAPLEY APPROACH
# def predict_shapley(model, tokenizer, toxic_mol):
#     """
#     Predict function for SHAP.
    
#     Args:
#     - model (torch.nn.Module): The PyTorch model to make predictions.
#     - tokenizer: The tokenizer used for encoding the input data.
#     - toxic_mol (list): List of molecules to predict.
    
#     Returns:
#     - numpy.ndarray: Logit values for the second class.
#     """
#     model.eval()  # Ensure the model is in evaluation mode
#     tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=128, truncation=True) for v in toxic_mol])
    
#     with torch.no_grad():  # Ensure no gradients are calculated
#         outputs = model(tv)[0].cpu().numpy()
#     scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
#     val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
#     # logger.info(f'SHAPLEY probs: {scores}')
#     # logger.info(f'SHAPLEY values: {val}')
#     return val

# def compute_shapley_values(model, tokenizer, toxic_mol):
#     """
#     Compute SHAP values for the given data.
    
#     Args:
#     - model (torch.nn.Module): The PyTorch model for which SHAP values are computed.
#     - tokenizer: The tokenizer used for encoding the input data.
#     - toxic_mol (list): List of molecules for which SHAP values are computed.
    
#     Returns:
#     - list: SHAP values.
#     """
#     # Masker for SHAP
#     masker = shap.maskers.Text(tokenizer)
#     # Build an explainer using the masker
#     explainer = shap.Explainer(lambda x: predict_shapley(model, tokenizer, x), masker)
#     shap_values = explainer([toxic_mol])    
#     return shap_values

# def shapley_approach(toxic_mol, tokenizer, classifier):
#     logger.info('SHAPLEY APPROACH IN PROGRESS...')
#     # Build an explainer using a token masker
#     # Ensure model is in evaluation mode
#     classifier.eval()
#     shapley_values = compute_shapley_values(classifier, tokenizer, toxic_mol)
#     values = shapley_values.values[0]
#     tokens = shapley_values.data[0]
#     # output the shapley values
#     shaps = (values, tokens)
#     # remove the special tokens from the fragments and corresponding shapley values
#     shaps = (values[1:-1], tokens[1:-1])
#     return shaps

# def augment_selfies(tr_text, tr_labels, augmentation_factor):
#     toxic_selfies = [s for s, l in zip(tr_text, tr_labels) if l == 1]
#     non_toxic_selfies = [s for s, l in zip(tr_text, tr_labels) if l == 0]

#     num_toxic = len(toxic_selfies)
#     num_non_toxic = len(non_toxic_selfies)
#     augmentations_needed = (num_non_toxic - num_toxic * augmentation_factor) // augmentation_factor
    
#     logger.info(f"Number of toxic SELFIES: {num_toxic}")
#     logger.info(f"Number of non-toxic SELFIES: {num_non_toxic}")
#     logger.info(f"Number of augmentations needed: {augmentations_needed}")
    
#     unique_augmented_selfies = set()

#     for _ in range(augmentations_needed):
#         selected_selfies = random.choice(toxic_selfies)
        
#         # Enumerate SELFIES before generating non-canonical versions
#         enumerated = enumerate_selfies(selected_selfies)
        
#         for enum_selfies in enumerated:
#             non_canonicals = generate_non_canonical_selfies(enum_selfies, augmentation_factor)
#             unique_augmented_selfies.update(non_canonicals)

#     # Convert set back to list and append
#     augmented_selfies = list(unique_augmented_selfies)
    
#     tr_text.extend(augmented_selfies)
#     tr_labels.extend([1] * len(augmented_selfies))
    
#     return tr_text, tr_labels


# def augment_smiles(tr_text, tr_labels, augmentation_factor):
#     # Filter out toxic and non-toxic smiles using list comprehensions
#     toxic_smiles = [s for s, l in zip(tr_text, tr_labels) if l == 1]
#     non_toxic_smiles = [s for s, l in zip(tr_text, tr_labels) if l == 0]
    
#     num_toxic = len(toxic_smiles)
#     num_non_toxic = len(non_toxic_smiles)
#     logger.info(f"Number of toxic SMILES: {num_toxic}")
#     logger.info(f"Number of non-toxic SMILES: {num_non_toxic}")

#     if num_non_toxic > num_toxic :
#         toxic_augmentations_needed = (num_non_toxic - num_toxic * augmentation_factor) // augmentation_factor
#         logger.info(f"Number of toxic_augmentations_needed: {toxic_augmentations_needed}")
          
#         # Set to ensure uniqueness of augmented SMILES
#         unique_augmented_smiles = set()
#         for _ in range(toxic_augmentations_needed):
#             selected_smiles = random.choice(toxic_smiles)
#             # Enumerate SMILES before generating non-canonical versions
#             enumerated = enumerate_smiles(selected_smiles)
#             for enum_smiles in enumerated:
#                 non_canonicals = generate_non_canonical_smiles(enum_smiles, num_variants=augmentation_factor)
#                 unique_augmented_smiles.update(non_canonicals)

#         # Convert set back to list and append
#         augmented_smiles = list(unique_augmented_smiles)
        
#         tr_text.extend(augmented_smiles)
#         tr_labels.extend([1] * len(augmented_smiles))
            

#     else :
#         non_toxic_augmentations_needed = (num_toxic - num_non_toxic * augmentation_factor) // augmentation_factor
#         logger.info(f"Number of {non_toxic_augmentations_needed}: {non_toxic_augmentations_needed}")

#         # Set to ensure uniqueness of augmented SMILES
#         unique_augmented_smiles = set()
#         for _ in range(non_toxic_augmentations_needed):
#             selected_smiles = random.choice(non_toxic_smiles)
#             # Enumerate SMILES before generating non-canonical versions
#             enumerated = enumerate_smiles(selected_smiles)
#             for enum_smiles in enumerated:
#                 non_canonicals = generate_non_canonical_smiles(enum_smiles, num_variants=augmentation_factor)
#                 unique_augmented_smiles.update(non_canonicals)

#         # Convert set back to list and append
#         augmented_smiles = list(unique_augmented_smiles)
#         tr_text.extend(augmented_smiles)
#         tr_labels.extend([0] * len(augmented_smiles))
  
    
    
#     return tr_text, tr_labels


# def augment_selfies(tr_text, tr_labels, augmentation_factor):
#     toxic_selfies = [s for s, l in zip(tr_text, tr_labels) if l == 1]
#     non_toxic_selfies = [s for s, l in zip(tr_text, tr_labels) if l == 0]

#     num_toxic = len(toxic_selfies)
#     num_non_toxic = len(non_toxic_selfies)
#     augmentations_needed = (num_non_toxic - num_toxic * augmentation_factor) // augmentation_factor
    
#     logger.info(f"Number of toxic SELFIES: {num_toxic}")
#     logger.info(f"Number of non-toxic SELFIES: {num_non_toxic}")
#     logger.info(f"Number of augmentations needed: {augmentations_needed}")
    
#     unique_augmented_selfies = set()

#     for _ in range(augmentations_needed):
#         selected_selfies = random.choice(toxic_selfies)
        
#         # Enumerate SELFIES before generating non-canonical versions
#         enumerated = enumerate_selfies(selected_selfies)
        
#         for enum_selfies in enumerated:
#             non_canonicals = generate_non_canonical_selfies(enum_selfies, augmentation_factor)
#             unique_augmented_selfies.update(non_canonicals)

#     # Convert set back to list and append
#     augmented_selfies = list(unique_augmented_selfies)
    
#     tr_text.extend(augmented_selfies)
#     tr_labels.extend([1] * len(augmented_selfies))
    
#     return tr_text, tr_labels



# def perturb_sample(sample, feature , tokenizer , method):
#     """
#     Perturbs the given sample by modifying the specified feature.
    
#     Parameters:
#     - sample: The original sample.
#     - feature: The feature to be perturbed.
    
#     Returns:
#     - perturbed_sample: The sample after perturbation.
#     - replaced_with: The fragment that replaced the original feature.
#     """
    
#     if (method == 'LIME' or method == 'shapley') and tokenizer.tokenizer_type=='WordPiece':
#         fragments = custom_tokenizer(sample, tokenizer)
#     else:
#         fragments = tokenizer.tokenize(sample)
        
#     if feature not in fragments:
#         logger.info(f' feature not in fragments : {feature}')
#         return sample, None  # Return the original sample if the feature is not found
    
#     # Create a list of unique fragments excluding the feature
#     unique_fragments = list(set(fragments) - {feature})
#     if not unique_fragments:
#         return sample, None  # Return the original sample if no alternative fragments are available
    
#     # Replace the feature with a random fragment from the unique fragments list
#     # select 2 fragments randomly
#     # Replace the feature with a random fragment from the unique fragments list
#     # select 2 fragments randomly
#     replaced_with1 = np.random.choice(unique_fragments)
#     potential_replacements = list(set(unique_fragments) - {replaced_with1})
#     if potential_replacements:
#         replaced_with2 = np.random.choice(potential_replacements)
#     else:
#         replaced_with2 = None
#     perturbed_sample = sample.replace(feature, replaced_with1)

#     # Log the original sample, feature, replaced fragment, and perturbed sample
#     logger.info(f"Original Sample: {sample}")
#     logger.info(f"Fragments: {fragments}")
#     logger.info(f"Feature: {feature}")
#     logger.info(f"Replaced With: {replaced_with1}") 
#     logger.info(f"Perturbed Sample: {perturbed_sample}")

#     # # return the perturbed sample as string 
#     perturb_sample = ' '.join(perturbed_sample)
#     replaced_with1 = ' '.join(replaced_with1)
#     if replaced_with2:
#         replaced_with2 = ' '.join(replaced_with2)
    
#     return perturb_sample, replaced_with2



    # # log in csv file
    # path = f'./logs/xai/predicted_mols_{args.tokenizer_type}_{args.mol_rep}_{args.target_data}.csv'
    # if not os.path.exists(path):
    #     df = pd.DataFrame(columns=['task','vocab','mols',
    #                                'tox_mols','toxic_mols_predicted_toxic','tox_prediction_ratio','toxic_mols_predicted_non_toxic', 
    #                                'non_tox_mols','non_toxic_mols_predicted_non_toxic','non_tox_prediction_ratio','non_toxic_mols_predicted_toxic',
    #                                ])
    #     df = df.append({'task': args.task, 'vocab': vocab_size, 'mols': len(mols),
    #                     'tox_mols': len(toxic_mols), 'toxic_mols_predicted_toxic': len(toxic_mols_predicted_toxic), 'tox_prediction_ratio': len(toxic_mols_predicted_toxic)/len(toxic_mols), 'toxic_mols_predicted_non_toxic': len(toxic_mols_predicted_non_toxic), 
    #                     'non_tox_mols': len(non_toxic_mols), 'non_toxic_mols_predicted_non_toxic': len(non_toxic_mols_predicted_non_toxic), 'non_tox_prediction_ratio': len(non_toxic_mols_predicted_non_toxic)/len(non_toxic_mols), 'non_toxic_mols_predicted_toxic': len(non_toxic_mols_predicted_toxic)}, ignore_index=True)
    #     df.to_csv(path, index=False)
    # else:
    #     df = pd.read_csv(path)
    #     df = df.append({'task': args.task, 'vocab': vocab_size, 'mols': len(mols),
    #                     'tox_mols': len(toxic_mols), 'toxic_mols_predicted_toxic': len(toxic_mols_predicted_toxic), 'tox_prediction_ratio': len(toxic_mols_predicted_toxic)/len(toxic_mols), 'toxic_mols_predicted_non_toxic': len(toxic_mols_predicted_non_toxic),
    #                     'non_tox_mols': len(non_toxic_mols), 'non_toxic_mols_predicted_non_toxic': len(non_toxic_mols_predicted_non_toxic), 'non_tox_prediction_ratio': len(non_toxic_mols_predicted_non_toxic)/len(non_toxic_mols), 'non_toxic_mols_predicted_toxic': len(non_toxic_mols_predicted_toxic)}, ignore_index=True)
    #     df.to_csv(path, index=False)




# data = pd.read_csv('./logs/xai/BPE_NR-AR-LBD_smiles_tox21_10000.csv')
#     toxic_mols = data['toxic_mols_predicted_toxic']
#     logger.info(f' length of toxic mols : {len(toxic_mols)}')
#     dictionary = {}
#     for toxic_mol in toxic_mols_predicted_toxic:
#         fragments = tokenizer.tokenize(toxic_mol)
#         counter= 0
#         for fragment in fragments:
#             mol = Chem.MolFromSmiles(fragment)
#             if mol is not None:
#                     # logger.info(f' fragment : {fragment} could be converted to mol')
#                     counter+=1

#         dictionary[toxic_mol] = counter
#         # sort the dictionary
#         dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
#         # save the dictionary
#         path = f'./logs/xai/fragments_toxic_mols_{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{args.vocab_size}.csv'
#         df= pd.DataFrame(dictionary.items(), columns=['toxic_mol', 'number_of_valid_fragments'])
#         df.to_csv(path, index=False)
#         logger.info(f' dictionary : {dictionary}')
        

    # ranked_molecules = rank_molecules_by_convertible_fragments(toxic_mols_predicted_toxic, tokenizer)
    # # save the ranked molecules in csv file
    # df = pd.DataFrame(ranked_molecules, columns=['molecule', 'percentage_graphable'])
    # df.to_csv(f'./logs/xai/ranked_molecules_{args.tokenizer_type}_{args.task}.csv', index=False)
