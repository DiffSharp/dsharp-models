// Copyright 2020 The TensorFlow Authors, adapted by the DiffSharp authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

open DiffSharp

/// NeuMF is a recommendation model that combines matrix factorization and a multi-layer perceptron.
///
/// Original Paper:
/// "Neural Collaborative Filtering"
/// Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua
/// https://arxiv.org/pdf/1708.05031.pdf

type NeuMF: Module {

    type Scalar = Float
    let numUsers: int
    let numItems: int
    let numLatentFeatures: int
    let matrixRegularization: scalar
    let mlpLayerSizes: int[] = [64, 32, 16, 8]
    let mlpRegularizations: scalar[] = [0, 0, 0, 0]

    let mfUserEmbedding: Embedding<Scalar>
    let mfItemEmbedding: Embedding<Scalar>

    let mlpUserEmbedding: Embedding<Scalar>
    let mlpItemEmbedding: Embedding<Scalar>
    let mlpLayers: [Dense<Scalar>]

    let neuMFLayer: Dense<Scalar>

    /// Initializes a NeuMF model as per the dataset from the given hyperparameters.
    ///
    /// -Parameters
    /// - numUsers: Total number of users in the dataset.
    /// - numItems: Total number of items in the dataset.
    /// - numLatentFeatures: Embedding size of the matrix factorization model.
    /// - matrixRegularization: Regularization for the matrix factorization embeddings.
    /// - mlpLayerSizes: The sizes of the layers in the multi-layer perceptron model.
    /// - mlpRegularizations: Regularization for each multi-layer perceptron layer.
    ///
    ///  Note: The first MLP layer is the concatenation of user and item embeddings, so mlpLayerSizes[0]/2 is the embedding size.
    public init(
        numUsers : int,
        numItems : int,
        numLatentFeatures : int,
        matrixRegularization : double,
        mlpLayerSizes : int[],
        mlpRegularizations : double[]
    ) = 

        Debug.Assert(mlpLayerSizes[0] % 2 = 0, "Input of first MLP layers must be multiple of 2")
        Debug.Assert(
            mlpLayerSizes.count = mlpRegularizations.count,
            "Size of MLP layers and MLP regularization must be equal")

        self.numUsers = numUsers
        self.numItems = numItems
        self.numLatentFeatures = numLatentFeatures
        self.matrixRegularization = matrixRegularization
        self.mlpLayerSizes = mlpLayerSizes
        self.mlpRegularizations = mlpRegularizations
        mlpLayers = []

        // TODO: regularization
        // Embedding Layer
        mfUserEmbedding = Embedding<Scalar>(
            vocabularySize=self.numUsers, embeddingSize=self.numLatentFeatures)
        mfItemEmbedding = Embedding<Scalar>(
            vocabularySize=self.numItems, embeddingSize=self.numLatentFeatures)
        mlpUserEmbedding = Embedding<Scalar>(
            vocabularySize=self.numUsers, embeddingSize=self.mlpLayerSizes[0] / 2)
        mlpItemEmbedding = Embedding<Scalar>(
            vocabularySize=self.numItems, embeddingSize=self.mlpLayerSizes[0] / 2)

        for (inputSize, outputSize) in zip(mlpLayerSizes, mlpLayerSizes[1..]) = 
            mlpLayers.append(Linear(inFeatures=inputSize, outFeatures=outputSize, activation= dsharp.relu))


        neuMFLayer = Linear(inFeatures=(self.mlpLayerSizes |> Array.last + self.numLatentFeatures), outFeatures=1)


    
    override _.forward(input: Tensor (*<int32>*)) : Tensor =
        // Extracting user and item from dataset
        let userIndices = input.unstacked(alongAxis: 1)[0]
        let itemIndices = input.unstacked(alongAxis: 1)[1]

        // MLP part
        let userEmbeddingMLP = mlpUserEmbedding(userIndices)
        let itemEmbeddingMLP = mlpItemEmbedding(itemIndices)

        // MF part
        let userEmbeddingMF = mfUserEmbedding(userIndices)
        let itemEmbeddingMF = mfItemEmbedding(itemIndices)

        let mfVector = userEmbeddingMF * itemEmbeddingMF

        let mlpVector = userEmbeddingMLP.cat(itemEmbeddingMLP, alongAxis: -1)
        mlpVector = mlpLayers.differentiableReduce(mlpVector){ $1($0)

        // Concatenate MF and MLP parts
        let vector = mlpVector.cat(mfVector, alongAxis: -1)

        // Final prediction layer
        return neuMFLayer(vector)


