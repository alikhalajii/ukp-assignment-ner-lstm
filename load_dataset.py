from torchtext.data import Dataset, Example

def load_dataset(filename, fields):
    #Load words and labels from the files, creating a TorchText dataset.

    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()[2:]
    
    examples = [] # List of examples
    words = []
    labels = []

    # Remove any whitespace characters from the beginning and end of each line
    for line in lines:
        line = line.strip()

        # If reached to end of sentence
        if not line:  
            examples.append(Example.fromlist([words, labels], fields))
            words = []
            labels = []

        # If current line is not empty,the line will be splitted into list of words.
        else:
            columns = line.split()
            words.append(columns[0])
            labels.append(columns[-1])
    
    # List of dictionaries containing 'text' and 'label' keys
    return Dataset(examples, fields)

