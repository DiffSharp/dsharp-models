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

#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\Library.dll"
#r "System.Runtime.Extensions.dll"
#r "System.Reflection.Emit.dll"
#r @"..\..\..\DiffSharp\tests\DiffSharp.Benchmarks.Python\bin\Release\netcoreapp3.1\Python.Runtime.dll"

open DiffSharp
open Python
open Python.Runtime
open FSharp.Reflection
open System

let iterationCount = 10000
let learningPhase = iterationCount * 5 / 100
let rnd = Random()

type Strategy = bool

type PyObject with 
    member t.toInt() = (t.AsManagedObject(typeof<int>) :?> int)
    member t.toBool() = (t.AsManagedObject(typeof<bool>) :?> bool)
    member t.toString() = (t.AsManagedObject(typeof<string>) :?> string)
    member t.toArb() : 'T = (t.AsManagedObject(typeof<'T>) :?> 'T)
    member t.toArbTuple3() : 'T1 * 'T2 * 'T3 =
        t.Item(0).toArb(), t.Item(1).toArb(), t.Item(2).toArb()

let (?) (pyobj: PyObject) (nm:string) : 'T = 
    let t = typeof<'T>
    let ds,r = 
        if FSharpType.IsFunction(t) then
            let d,r = FSharpType.GetFunctionElements(t)
            let els = if typeof<unit> = d then [| |] elif Reflection.FSharpType.IsTuple(d) then Reflection.FSharpType.GetTupleElements(d) else [| d |]
            els, r
        else
            [| |], t
    let postConv = 
        if r = typeof<PyObject> then 
            (fun (a: PyObject) -> box a)
        else (fun (a: PyObject) -> a.AsManagedObject(r))
    match ds with 
    | [| |] -> pyobj.InvokeMethod(nm) |> box |> unbox
    | [| d1 |] -> 
        printfn "aaaaaaaaaaa"
        FSharpValue.MakeFunction(t, (fun arg -> pyobj.InvokeMethod(nm, arg.ToPython()) |> postConv)) |> unbox
    | _ -> 
        FSharpValue.MakeFunction(t, (fun arg -> let args = FSharpValue.GetTupleFields(arg) in pyobj.InvokeMethod(nm, [| for arg in args -> arg.ToPython() |]) |> postConv)) |> unbox

let gil = Py.GIL()
//let scope = Py.CreateScope()
// https://gym.openai.com/docs/
// pip install gym

let gym = Py.Import("gym")
let environment : PyObject = gym?make("Blackjack-v0")

    //let gym = Python.import("gym")
//let environment = gym.make("Blackjack-v0")

type BlackjackState(pythonState: PyObject) =
    member _.playerSum : int = pythonState?playerSum
    member _.dealerCard: int = pythonState?dealerCard
    member _.useableAce: int = pythonState?useableAce

type SolverType =
    | Random
    | Markov
    | Qlearning
    | Normal
    static member allCases = [Random; Markov; Qlearning; Normal]

type Solver() =
    let playerStateCount = 32 // 21 + 10 + 1 offset
    let dealerVisibleStateCount = 11 // 10 + 1 offset
    let aceStateCount = 2 // useable / not bool
    let playerActionCount = 2 // hit / stay
    let Q = Array.init playerStateCount (fun i -> 
              Array.init dealerVisibleStateCount (fun j -> 
                 Array.init aceStateCount (fun _ -> 
                    Array.init playerActionCount (fun _ -> 0.0))))
    let alpha: double = 0.5
    let gamma: double = 0.2

    let randomStrategy() =
        rnd.NextDouble() < 0.5

    let markovStrategy(observation: BlackjackState) =
        // hit @ 80% probability unless over 18, in which case do the reverse
        let flip = rnd.NextDouble()
        let threshHold: double = 0.8

        if observation.playerSum < 18 then
            flip < threshHold
        else
            flip > threshHold

    let normalStrategyLookup(playerSum: int) =
        // see figure 11: https://ieeexplore.ieee.org/document/1299399/
        match playerSum with
        | 10 -> "HHHHHSSHHH"
        | 11 -> "HHSSSSSSHH"
        | 12 -> "HSHHHHHHHH"
        | 13 -> "HSSHHHHHHH"
        | 14 -> "HSHHHHHHHH"
        | 15 -> "HSSHHHHHHH"
        | 16 -> "HSSSSSHHHH"
        | 17 -> "HSSSSHHHHH"
        | 18 -> "SSSSSSSSSS"
        | 19 -> "SSSSSSSSSS"
        | 20 -> "SSSSSSSSSS"
        | 21 -> "SSSSSSSSSS"
        | _ -> "HHHHHHHHHH"

    let normalStrategy(observation: BlackjackState) =
        if observation.playerSum = 0 then
            true
        else
            let lookupString = normalStrategyLookup(observation.playerSum)
            lookupString.[observation.dealerCard - 1] = 'H'

    let qLearningStrategy(observation: BlackjackState, iteration: int) : Strategy =
        let qLookup = Q.[observation.playerSum].[observation.dealerCard].[observation.useableAce]
        let stayReward = qLookup.[0]
        let hitReward = qLookup.[1]

        if iteration < rnd.Next(learningPhase) + 1 then
            randomStrategy()
        elif iteration > learningPhase then
            // quit learning after initial phase
            alpha = 0.0 
        elif hitReward = stayReward then
            randomStrategy()
        else
            hitReward > stayReward

    member _.strategy(observation: BlackjackState, solver: SolverType, iteration: int) =
        match solver with
        | Random ->
            randomStrategy()
        | Markov ->
            markovStrategy(observation)
        | Qlearning ->
            qLearningStrategy(observation, iteration)
        | Normal ->
            normalStrategy(observation)

    member _.updateQLearningStrategy(prior: BlackjackState, action: int, reward: int, post: BlackjackState) = 
        let oldQ = Q.[prior.playerSum].[prior.dealerCard].[prior.useableAce].[action]
        let priorQ = (1. - alpha) * oldQ

        let maxReward = max(Q.[post.playerSum].[post.dealerCard].[post.useableAce].[0])
                           (Q.[post.playerSum].[post.dealerCard].[post.useableAce].[1])
        let postQ = alpha * (double reward + gamma * maxReward)

        Q.[prior.playerSum].[prior.dealerCard].[prior.useableAce].[action] <- oldQ + priorQ + postQ // check me, oldQ twice??


let learner = Solver()

let solver = SolverType.Random

//for solver in SolverType.allCases do
let mutable totalReward = 0
let i = 1
//    for i in 1..iterationCount do
let mutable isDone = false
environment.InvokeMethod("reset") |> ignore

//        while not isDone do
let priorState = BlackjackState(pythonState=environment?_get_obs())
let action: int = 
    if learner.strategy(observation=priorState, solver=solver, iteration=i) then 1 else 0

let (pythonPostState: PyObject, reward: int, finished: bool) = environment?step(action)

if solver = Qlearning then
    let postState = BlackjackState(pythonState=pythonPostState)
    learner.updateQLearningStrategy(prior=priorState,
                                    action=action,
                                    reward=reward (* int(reward) ?? 0 *),
                                    post=postState)


if finished then
    totalReward <- totalReward + reward (* int(reward) ?? 0 *)
    isDone <- true

printfn $"Solver: {solver}, Total reward: {totalReward} / {iterationCount} trials"

