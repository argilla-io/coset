from torchtext import data

DATASET_PATH = '/Users/oeg/dev/recognai/data/datasets/cosetdata/'

# Primero definimos los campos del dataset de entrenamiento
twitter_id = data.Field()
text = data.Field()
label = data.Field(use_vocab=False, sequential=False)

train = data.TabularDataset(path= DATASET_PATH + 'coset-train.csv',
                            format='csv',
                            fields= [('id', twitter_id), ('text', text),('label',label)]
                           )
for i, e in enumerate(train.examples):
    print(i, e.text)
