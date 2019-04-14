# Reproducability Challenge
This repo contains code reproducing results of ["Leveraging uncertainty information from deep neural networks for disease detection"](https://www.nature.com/articles/s41598-017-17876-z])

Original code used in paper:

<https://github.com/chleibig/disease-detection>

Paper utilizes a pre-trained model described in this blogpost:

<http://jeffreydf.github.io/diabetic-retinopathy-detection/>

# The Process

Before any training is done, the images are preprocessed as a good heuristic.
This is effectively like appending an extra layer called "expert advice" behind
the neural network.
