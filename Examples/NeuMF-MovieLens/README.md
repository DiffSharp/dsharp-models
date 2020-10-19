# TopK Recommendation System using Neural Collaborative Filtering

This example demonstrates how to train a recommendation system with implicit feedback on the
MovieLens 100K (ml-100k) dataset using a [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
model. This model trains on binary information about whether or not a user interacted with a specific item.
To target the models for an implicit feedback and ranking task, we optimize them using sigmoid cross entropy
loss with negative sampling.

## Setup

