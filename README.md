# Coreference Resolution CNN
## Project Structure
- / (root): Contain executable python files (create embedding, extract feature) and training/testing jupyter notebooks.
    - data/: Contains training and testing data.
    - helper_files/: Contains files (not code) that is needed by the system e.g. word embedding
    - model builders/: Builder class for singleton classifier and coreference classifier model
    - models/: Trained singleton classifier and coreference classifier models.
    - tensor_builders/: Helper file for building Keras tensors.
    - utils/: Helper codes e.g. data structures, coreference scorers, markable clusterers. trainining instance generator, etc.