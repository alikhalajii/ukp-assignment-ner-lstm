import torch
from torchtext.data import Field, BucketIterator
from torchtext.vocab import Vectors
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from load_dataset import load_dataset
from compute_class_weights import compute_class_weights
from f1_score import f1_score
from bilstm_tagger import BiLSTMTagger
from sequence_evaluation import evaluate


# Setup device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Using device: {device}")

# Configuration
data_path = "data/"
embedding_path = "data/glove.6B.50d.txt"
learning_rate = 0.001
weight_decay = 1e-5
epochs = 20
batch_size = 1

# For saving the final model
model_save_path = model_save_path = "models/ner_bilstm_tagger.pth"

def main():
    """
    Main function for training, validating and evaluating the BiLSTMTagger model.
    """
    
    # Define fields for input text and corresponding labels (the sequence of tokens, and the sequence of labels)
    text_field = Field(init_token='<bos>', eos_token='<eos>', is_target=False, sequential=True)
    label_field= Field(init_token='<bos>', eos_token='<eos>',is_target=True, sequential=True, unk_token=None)
    # Combine fields into a list for dataset loading
    fields = [('text', text_field), ('label', label_field)]

    # Load the training, validation, and test datasets using the 'load_dataset' function
    train_data = load_dataset(data_path + 'train.conll', fields)
    dev_data = load_dataset(data_path + 'dev.conll' , fields)
    test_data = load_dataset(data_path + 'test.conll', fields)

    # Calculate the total number of tokens and the number of sentences in the training dataset
    num_tokens_train = sum(len(example.text) + 2 for example in train_data)
    num_sentences_train = len(train_data)

    n_batches = num_sentences_train
    mean_n_tokens = num_tokens_train / n_batches

    # Use GLOVE embeddings to build the vocabulary
    text_field.build_vocab(train_data, vectors=Vectors(embedding_path))
    label_field.build_vocab(train_data)

    # Generate batches for training, validation, and test datasets. 
    # Sorting examples by sentence length minimizes padding, optimizing the training process for sequence models.
    train_iterator = BucketIterator(dataset=train_data, batch_size=batch_size, sort_key=lambda x: len(x.text), device=device, train=True, shuffle=True, sort=True)
    dev_iterator = BucketIterator(dataset=dev_data, batch_size=batch_size, sort_key=lambda x: len(x.text), device=device, train=False, sort=True)
    test_iterator = BucketIterator(dataset=test_data, batch_size=batch_size, sort_key=lambda x: len(x.text), device=device, train=False, sort=True)
   
    # Compute class weights based on the frequency of each label in the training dataset
    class_weights = compute_class_weights(data_path + 'train.conll')
    
    # Instantiate the BiLSTMTagger class and create a model object
    model = BiLSTMTagger(text_field, label_field, class_weights=class_weights)
    # Move the model object to the specified device
    model.to(device)

    # Create txhe Adam optimizer with the current hyperparameters
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

    # Set up a learning rate scheduler that uses the parameters found by hyperparameter fine-tuning.
    scheduler = StepLR(optimizer, step_size=5, gamma=0.2)

    # Setup early stopping
    best_f1 = 0.0
    patience_limit = 4
    patience_counter = 0

    result = defaultdict(list)
    result['val_f1'] = [] # Initialize 'f1' in result

    print(f"\n Training with learning rate: {learning_rate}, weight decay: {weight_decay}, with StepLR scheduler (step_size={scheduler.step_size}, gamma={scheduler.gamma})")
    print('\n Model training begins... \n')

    # Training Loop
    for epoch in range(1, epochs + 1):
        train_loss_sum = 0
        model.train()
        stats = defaultdict(Counter)
        for batch in list(train_iterator):
            loss = model(batch.text, batch.label) / mean_n_tokens
        
            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()
        
            train_loss_sum += loss.item()
        train_loss = train_loss_sum / n_batches
        result['train_loss'].append(train_loss)

        # Evaluate on the validation set (dev data)
        val_loss_sum = 0
        model.eval()
        with torch.no_grad():
            stats = defaultdict(Counter)
            for batch in list(dev_iterator):
                predicted = model.predict(batch.text)
                loss = model(batch.text, batch.label) / mean_n_tokens
                val_loss_sum += loss.item()
                evaluate(predicted, batch.label, label_field, stats)

            val_loss = val_loss_sum / len(dev_iterator)
            val_all_class_f1 = [f1_score(stats[i + 1]) for i, name in enumerate(label_field.vocab.itos[1:])]
            val_average_class_f1 = np.mean(val_all_class_f1)

        result['val_loss'].append(val_loss)
        result['val_f1'].append(val_average_class_f1)

        # Print out the training loss, validation loss and F1-Score on dev-data
        print(f'Epoch {epoch: >2}  |  Learning rate: {format(optimizer.param_groups[0]["lr"], ".0e")}  |  Training loss = {train_loss:.4f}  |  Validiation loss = {val_loss:.4f}  |  Macro-averaged F1 score on dev data = {val_average_class_f1:.4f}')  

        # Step the learning rate scheduler
        scheduler.step()

        # Check if F1 score has improved
        if val_average_class_f1 > best_f1:
            best_f1 = val_average_class_f1
            patience_counter = 0

            # Save the model if it has the best F1 score
            best_model_state = model.state_dict()
            torch.save({
                'model_state_dict': best_model_state, 
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'emb_dim': model.embedding.embedding_dim,
                'num_layers': model.lstm.num_layers,
                'hidden_size': model.lstm.hidden_size,
                'class_weights': class_weights,
                }, model_save_path)

        else:
            patience_counter += 1
            # Stop training if there is no improvement for a certain number of epochs
            if patience_counter >= patience_limit:
                print(f'\nEarly stopping at epoch {epoch} as there is no improvement on validation set.\n')
                break
    else:
        print(f'\nEarly stopping not triggered. Best validation F1-score: {best_f1:.4f}')
     
    print(f"Saving model to: {model_save_path}\n")

    
    # Put the model in evaluation mode on test data
    model.eval()
    with torch.no_grad():
        stats = defaultdict(Counter)
        for batch in list(test_iterator):
            predicted = model.predict(batch.text)        
            # evaluation of the predicted labels           
            evaluate(predicted, batch.label, label_field, stats)

    # Compute F1 score for each class
    all_class_f1 = []
    class_names = [class_name for class_name in label_field.vocab.itos[1:]]
    for class_id, class_name in enumerate(class_names):
            class_stats = stats[class_id + 1]  # Adjusting for 1-based index
            class_f1 = f1_score(class_stats)
            all_class_f1.append(class_f1)
    # Compute and print the average F1 score across all classes
    average_f1 = np.mean(all_class_f1)
    print(f'\033[1mMacro-averaged F1 scores of the final model on the test data: {average_f1:.4f}\033[0m \n')

    # Plot the curves
    plt.plot(result['train_loss'])
    plt.plot(result['val_loss'])
    plt.plot(result['val_f1'])

    plt.xlabel("Epochs")
    plt.xticks(np.arange(0, epochs + 1, step=2))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.legend(['Training Loss', 'Validation Loss (dev data)', 'Macro F1-score (dev data)'])
    plt.show()

if __name__ == "__main__":
    main()