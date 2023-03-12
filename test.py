import os
import shutil

from tensorflow.keras import layers
from tensorflow.keras import losses

import training

# Leggo il dataset da un URL
dataset = training.download_file(
    # Nome del file
    fname="aclImdb_v1",

    # URL di download
    origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",

    # Posto a `True`, consente di scompattare l'archivio
    untar=True
)

# Recupera il path del dataset e ci aggiunge in coda `aclImdb`
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

# Scende di un livello nella directory `aclImdb` e va dentro `train`
train_dir = os.path.join(dataset_dir, 'train')

# Rimuove la directory `unsup` da dentro a `train`
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# Definisce la dimensione dei batch (i piccoli lotti di dati da gestire per l'allenamento - al termine di ogni batch,
# procede a effettuare gli opportuni adattamenti)
batch_size = 32
seed = 42

raw_train_ds = training.training_dataset_from_directory(
    # Directory da cui vengono presi i dati. Seguono le note della documentazione:
    #
    #       directory: Directory where the data is located.
    #       If `labels` is "inferred", it should contain
    #       subdirectories, each containing text files for a class.
    #       Otherwise, the directory structure is ignored.
    directory='aclImdb/train',
    # Dimensione del batch
    batch_size=batch_size,
    # Frazione opzionale di dati da conservare per la validazione
    validation_split=0.2,
    seed=seed
)

# Dataset per la validazione
raw_val_ds = training.validation_dataset_from_directory(
    directory="aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    seed=seed
)

# Dataset contenente i test da usare per valutare le prestazioni del modello
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)

embedding_dim = 16

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=training.custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
test_text = raw_test_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Stampa l'architettura del modello
model.summary()
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.optimizers.Adam.__name__,
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Preparo i dati di validazione e test
val_ds = raw_val_ds.map(vectorize_text)
train_ds = raw_train_ds.map(vectorize_text)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10)

# Preparo i dati di test per valutare il modello
test_ds = raw_test_ds.map(vectorize_text)
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Esporto il modello in modo tale da potergli far "digerire" le stringhe
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

res = export_model.predict(["The film was the worst thing I've ever seen"])
print(res)
