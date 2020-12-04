// Copyright 2019 The TensorFlow Authors, adapted by the DiffSharp authors. All Rights Reserved.
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
#r @"..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\bin\Debug\netcoreapp3.1\publish\Library.dll"
#r "System.Runtime.Extensions.dll"

open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Util

// OLD NOTE:
// Note: This is a work in progress and training doesn't quite work.
// Here are areas for improvement:
// - Adopt a more principled reinforcement learning algorithm (e.g. policy
//   gradients). The algorithm should perform some tensor computation (not a
//   purely table-based approach).

type Observation = Tensor
type Reward = float32

let rng = RandomNumberGenerator()
type CatchAction =
    | None = 0
    | Left = 1
    | Right = 2


type CatchAgent(learningRate, initialReward) =
    //interface Agent
    //type Action = CatchAction

    let model = 
        Linear(inFeatures=3, outFeatures=50) 
        --> dsharp.sigmoid 
        --> Linear(inFeatures=50, outFeatures=3)
        --> dsharp.sigmoid

    let learningRate: double = learningRate
    let optimizer = Adam(model, dsharp.tensor(learningRate))
    let mutable previousReward = initialReward

    /// Performs one "step" (or parameter update) based on the specified
    /// observation and reward.
    member _.step(observation: Observation, reward: Reward) = 
        previousReward <- reward

        let x = dsharp.tensor(observation).unsqueeze(0)
        let (ŷ, backprop) = model.appliedForBackpropagation(x)
        let maxIndex = ŷ.argmax().[0]

        let δloss = -log(dsharp.tensor(ŷ.max(), dtype=Dtype.Float32)).expand( ŷ.shape) * previousReward
        let (δmodel, _) = backprop(δloss)
        optimizer.step()

        enum<CatchAction>(int(maxIndex))

    /// Returns the perfect action, given an observation.
    /// If the ball is left of the paddle, returns `left`.
    /// If the ball is right of the paddle, returns `right`.
    /// Otherwise, returns `none`.
    ///
    /// - Note: This function is for reference and is not used by `CatchAgent`.
    member _.perfectAction(observation: Observation) = 
        let paddleX = observation.[0].toScalar() :?> float32
        let ballX = observation.[1].toScalar() :?> float32
        if paddleX = ballX then CatchAction.None
        elif paddleX < ballX then CatchAction.Left
        else CatchAction.Right

    member _.Model = model

type Position = { mutable x: int; mutable y: int }

type CatchEnvironment(rowCount: int, columnCount: int) =
    let rowCount = rowCount
    let columnCount = columnCount
    let mutable ballPosition = { x = 0; y =0 }
    let mutable paddlePosition = { x = 0; y =0 }
    let action = CatchAction.None


    /// If the ball is in the bottom row:
    /// - Returns 1 if the horizontal distance from the ball to the paddle is
    ///   less than or equal to 1.
    /// - Otherwise, returns -1.
    /// If the ball is not in the bottom row, returns 0.
    let reward : Reward =
        if ballPosition.y = rowCount then
            if abs(ballPosition.x - paddlePosition.x) <= 1 then 1.0f else -1.0f
        else 0.0f

    /// Returns an obeservation of the game grid.
    let observation =
        dsharp.tensor ([double(ballPosition.x) / double(columnCount),  
                        double(ballPosition.y) / double(rowCount),
                        double(paddlePosition.x) / double(columnCount)])

    /// Returns the game grid as a 2D matrix where all scalars are 0 except the
    /// positions of the ball and paddle, which are 1.
    let grid = 
        dsharp.init2d rowCount columnCount (fun i j -> 
            if j = ballPosition.y && i = ballPosition.x then 1.0
            elif j = paddlePosition.y && i = paddlePosition.x then 1.0
            else 0.0)

    /// Resets the ball to be in a random column in the first row, and resets
    /// the paddle to be in the middle column of the bottom row.
    let reset() = 
        let randomColumn = rng.Next(0, columnCount)
        ballPosition <- { x=randomColumn; y=0 }
        paddlePosition <- { x=columnCount / 2; y=rowCount - 1 }
        observation

    do reset() |> ignore

    member _.step(action) = 
        // Update state.
        match action with
        | CatchAction.Left when paddlePosition.x > 0 ->
            paddlePosition.x <- paddlePosition.x - 1
        | CatchAction.Right when paddlePosition.x < columnCount - 1 ->
            paddlePosition.x <- paddlePosition.x + 1
        | _ ->
            ()

        ballPosition.y <- ballPosition.y + 1
        // Get reward.
        let currentReward = reward
        // Return observation and reward.
        if ballPosition.y = rowCount then
            (reset(), currentReward)
        else
            (observation, currentReward)

    member env.run() = 
        let mutable winCount = 0
        let mutable totalWinCount = 0
        let agent = CatchAgent(initialReward=reward, learningRate=dsharp.scalar 0.05)

        // Setup environment and agent.
        agent.Model.mode <- Mode.Train
        let maxIterations = 5000
        let mutable gameCount = 0
        while gameCount < maxIterations do
            let (observation, reward) = env.step(action)
            let action = agent.step(observation=observation, reward=reward)

            if reward <> 0.0f then
                gameCount <- gameCount + 1
                if reward > 0.0f then
                    winCount <- winCount + 1
                    totalWinCount <- totalWinCount + 1

                if gameCount % 20 = 0 then
                    print("Win rate (last 20 games): \(double(winCount) / 20)")
                    print($"""
                          Win rate (total): \
                          \(double(totalWinCount) / double(gameCount)) \
                          [{totalWinCount}/{gameCount}]
                          """)
                    winCount <- 0

        print($"""
              Win rate (final): \(double(totalWinCount) / double(gameCount)) \
              [{totalWinCount}/{gameCount}]
              """)

let env = CatchEnvironment(rowCount=5, columnCount=5)
