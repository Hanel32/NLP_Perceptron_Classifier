# NLP_Perceptron_Classifier
A perceptron classifier programmed in Python that trains on data iteratively and scores documents based upon their sentiment.
Please see "MaxEnt_and_Perceptron.pdf" for detailed implementation notes.

Most of the cool information has been written in the MaxEnt repository.

Essentially, this Perceptron classifier trains on documents based upon their sentiment. In training, a movie review is either marked as positive or negative manually. As the perceptron trains to different documents, weights for individual words that occur in the document are updated based upon the classification of the document and the occurence of the word. Because the perceptron is a much more low level classifier than MaxEnt, its only hyperparameter is the number of iterations utilized to fit to each document, rather than fitting to some epsilon, which denoted the average change across all documents.

What's interesting is, even though there is no gradient descent implemented, the Perceptron classifier is almost as accurate as the MaxEnt, being closer to the 72-74 percent accuracy area across 10 test splits.

To reproduce this experiment, check out the detailed implementation notes.

Enjoy!
