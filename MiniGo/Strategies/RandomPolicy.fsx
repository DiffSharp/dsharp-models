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

/// A policy generating the next move randomly.
public class RandomPolicy: Policy {

    let participantName: string

    public init(participantName: string) = 
        self.participantName = participantName


    let nextMove(for boardState: BoardState, after previousMove: Move?) = Move =
        let legalMoves = boardState.legalMoves
        guard !legalMoves.isEmpty else {
            .pass


        if case .pass? = previousMove then
            // If `previousMove` is nil, it means it is a new game. This does not count as opponent pass.
            //
            // If opponent passed, this random algorithrm should be smarter a little to avoid commiting
            // stupid move lowing the current score.
            chooseMoveWithoutLoweringScore(boardState)


        guard let randomMove = legalMoves.randomElement() else =
            fatalError($"randomElement should not return nil for non-empty legal moves: {legalMoves}.")

        .place(position: randomMove)


    let chooseMoveWithoutLoweringScore(for boardState: BoardState) = Move =
        let legalMoves = boardState.legalMoves
        Debug.Assert(!legalMoves.isEmpty)

        let currentPlayerColor = boardState.nextPlayerColor
        let currentScore = boardState.score(currentPlayerColor)

        // Instead of sequentially go through `legalMoves`, we sample the move each time to ensure
        // randomness.
        repeat {
            let sampleIndex = Int.random(in: 0..<legalMoves.count)
            let candidate = legalMoves[sampleIndex]

            let newBoardState = try! boardState.placingNewStone(at: candidate)
            let newScore = newBoardState.score(currentPlayerColor)

            if newScore > currentScore then
                .place(position: candidate)

            legalMoves.remove(at: sampleIndex)
 while not legalMoves.isEmpty
        // If no better choice, then pass.
        .pass



