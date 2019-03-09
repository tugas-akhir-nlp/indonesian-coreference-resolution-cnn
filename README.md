# Coreference Resolution CNN
## Project Structure
- / (root): Contain executable python files (create embedding, extract feature) and training/testing jupyter notebooks.
    - data/: Contains training and testing data.
    - helper_files/: Contains files (not code) that is needed by the system e.g. word embedding
    - model builders/: Builder class for singleton classifier and coreference classifier model
    - models/: Trained singleton classifier and coreference classifier models.
    - tensor_builders/: Helper file for building Keras tensors.
    - utils/: Helper codes e.g. data structures, coreference scorers, markable clusterers, 
    training instance generator, etc.

## Git LFS
This repository uses [Git Large File System (LFS)](https://git-lfs.github.com/) to store (big) files inside `data/` and `models/` folder.
To clone this repository, you should install [Git LFS](https://git-lfs.github.com/) first. After that, you can clone
this repository just like usual using `git clone`. Alternatively, you can use `git lfs clone` command to parallelize
the downloads and reduce the download time. For further information, you can check 
[Cloning an existing Git LFS repository](https://www.atlassian.com/git/tutorials/git-lfs#clone-respository).
