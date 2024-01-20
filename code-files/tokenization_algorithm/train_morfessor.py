import morfessor
import pickle
def save_model(model,model_path_name):
    # pickle.dump(arab_morfessor_tokenizer, open('./models/morfessor_models/arab/'+ arab_morf_name,'wb'))
    pickle.dump(model, open(model_path_name,'wb'))
    

io = morfessor.MorfessorIO()

train_data = list(io.read_corpus_file('../datasets/clintox.smi'))

# Create a new Morfessor model
model = morfessor.BaselineModel()

# Load your data into the model
model.load_data(train_data)

# Train the model using batch training
model.train_batch(max_epochs=1)

# save model
save_model(model , '../models/tokenizers/morfessors/morf_smilesDB_100.bin')