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

type IllegalMove: Error {
    case suicide
    case occupied

    /// A `ko` fight is a tactical and strategic phase that can arise in the game
    /// of go.
    ///
    /// The existence of ko fights is implied by the rule of ko, a special rule of
    /// the game that prevents immediate repetition of position, by a short 'loop'
    /// in which a single stone is captured, and another single stone immediately
    /// taken back.
    ///
    /// See https://en.wikipedia.org/wiki/Ko_fight for details.
    case ko


private enum PositionStatus: Equatable {
    case legal
    case illegal(reason: IllegalMove)


/// Represents an immutable snapshot of the current board state.
///
/// `BoardState` checks whether a new placed stone is legal or not. If so,
/// creates a new snapshot.
type BoardState {
    /// The game configuration.
    let gameConfiguration: GameConfiguration
    /// The color of the next player.
    let nextPlayerColor: Color

    /// The position of potential `ko`. See `IllegalMove.ko` for details.
    let ko: Position?

    /// All legal position to be considered as next move given the current board state.
    let legalMoves: [Position]

    /// All stones on the current board.
    let board: Board

    /// History of the previous board states (does not include current one).
    ///
    /// The most recent one is placed at index 0. The history count is truncated by
    /// `GameConfiguration.maxHistoryCount`.
    ///
    /// TODO(xiejw): Improve the efficient of history.
    let history: [Board]

    // General statistic.
    let playedMoveCount: int
    let stoneCount: int

    // Internal maintained states.
    let libertyTracker: LibertyTracker

    /// Constructs an empty board state.
    init(gameConfiguration: GameConfiguration) = 
        self.init(
            gameConfiguration: gameConfiguration,
            nextPlayerColor: .black,  // First player is always black.
            playedMoveCount: 0,
            stoneCount: 0,
            ko: nil,
            history: [],
            board: Board(size: gameConfiguration.size),
            libertyTracker: LibertyTracker(gameConfiguration: gameConfiguration)
        )


    private init(
        gameConfiguration: GameConfiguration,
        nextPlayerColor: Color,
        playedMoveCount: int,
        stoneCount: int,
        ko: Position?,
        history: [Board],
        board: Board,
        libertyTracker: LibertyTracker
    ) = 
        self.gameConfiguration = gameConfiguration
        self.nextPlayerColor = nextPlayerColor
        self.playedMoveCount = playedMoveCount
        self.stoneCount = stoneCount
        self.ko = ko

        assert(history.count <= gameConfiguration.maxHistoryCount)
        self.history = history

        self.libertyTracker = libertyTracker
        self.board = board
        precondition(board.size = gameConfiguration.size)

        if stoneCount = gameConfiguration.size * gameConfiguration.size then
            // Full board.
            self.legalMoves = []
        else
            self.legalMoves = board.allLegalMoves(
                ko: ko,
                libertyTracker: libertyTracker,
                nextPlayerColor: nextPlayerColor
            )



    /// Returns a new `BoardState` after current player passed.
    let passing() = BoardState {
        let newHistory = self.history
        newHistory.insert(self.board, at: 0)
        if newHistory.count > gameConfiguration.maxHistoryCount then
            _ = newHistory.popLast()

        return BoardState(
            gameConfiguration: self.gameConfiguration,
            nextPlayerColor: self.nextPlayerColor.opponentColor,
            playedMoveCount: self.playedMoveCount + 1,
            stoneCount: self.stoneCount,
            ko: nil,  // Reset ko.
            history: newHistory,
            board: self.board,
            libertyTracker: self.libertyTracker
        )


    /// Returns a new `BoardState` after placing a new stone at `position`.
    let placingNewStone(at position: Position) -> BoardState {
        // Sanity Check first.
        if case .illegal(let reason) = board.positionStatus(
            at: position,
            ko: self.ko,
            libertyTracker: self.libertyTracker,
            nextPlayerColor: self.nextPlayerColor
        ) = 
            throw reason


        // Gets copies of libertyTracker and board. Updates both by placing new stone.
        let currentStoneColor = self.nextPlayerColor
        let newLibertyTracker = self.libertyTracker
        let newBoard = self.board

        // Makes attempt to guess the possible ko.
        let isPotentialKo = newBoard.isKoish(at: position, withNewStoneColor: currentStoneColor)

        // Updates libertyTracker and board by placing a new stone.
        let capturedStones = try newLibertyTracker.addStone(at: position, withColor: currentStoneColor)
        newBoard.placeStone(at: position, withColor: currentStoneColor)

        // Removes capturedStones
        for capturedStone in capturedStones do
            newBoard.removeStone(at: capturedStone)


        // Updates stone count on board.
        let newStoneCount = self.stoneCount + 1 - capturedStones.count

        let newKo: Position?
        if let stone = capturedStones.first, capturedStones.count = 1, isPotentialKo then
            newKo = stone


        let newHistory = self.history
        newHistory.insert(self.board, at: 0)
        if newHistory.count > gameConfiguration.maxHistoryCount then
            _ = newHistory.popLast()


        return BoardState(
            gameConfiguration: self.gameConfiguration,
            nextPlayerColor: currentStoneColor = .black ? .white : .black,
            playedMoveCount: self.playedMoveCount + 1,
            stoneCount: newStoneCount,
            ko: newKo,
            history: newHistory,
            board: newBoard,
            libertyTracker: newLibertyTracker
        )


    /// Returns the score of the player.
    let score(for playerColor: Color) =
        let scoreForBlackPlayer = self.board.scoreForBlackPlayer(komi: self.gameConfiguration.komi)
        match playerColor with
        | .black ->
            return scoreForBlackPlayer
        | .white ->
            return -scoreForBlackPlayer




extension BoardState: CustomStringConvertible {
    let description: string {
        return board.description



extension BoardState: Equatable {
    public static let = (lhs: BoardState, rhs: BoardState) = Bool {
        // The following line is the sufficient and necessary condition for "equal".
        return lhs.board = rhs.board &&
            lhs.nextPlayerColor = rhs.nextPlayerColor &&
            lhs.ko = rhs.ko &&
            lhs.history = rhs.history



extension Board {
    /// Calculates all legal moves on board.
    let allLegalMoves(
        ko: Position?,
        libertyTracker: LibertyTracker,
        nextPlayerColor: Color
    ) = [Position] {
        let legalMoves = Array<Position>()
        for x in 0..<self.size {
            for y in 0..<self.size {
                let position = Position(x: x, y: y)
                guard .legal = positionStatus(
                    at: position,
                    ko: ko,
                    libertyTracker: libertyTracker,
                    nextPlayerColor: nextPlayerColor
                    ) else {
                        continue


                legalMoves.append(position)


        return legalMoves


    /// Checks whether a move is legal. If isLegal is false, reason will be set.
    let positionStatus(
        at position: Position,
        ko: Position?,
        libertyTracker: LibertyTracker,
        nextPlayerColor: Color
    ) = PositionStatus {
        guard self.color(at: position) = nil else { return .illegal(reason: .occupied)
        guard position <> ko else { return .illegal(reason: .ko)

        guard !isSuicidal(
            at: position,
            libertyTracker: libertyTracker,
            nextPlayerColor: nextPlayerColor
            ) else {
                return .illegal(reason: .suicide)

        return .legal


    /// A fast algorithm to check a possible suicidal move.
    ///
    /// This method assume the move is not `ko`.
    let isSuicidal(
        at position: Position,
        libertyTracker: LibertyTracker,
        nextPlayerColor: Color
    ) = Bool {
        let possibleLiberties = Set<Position>()

        for neighbor in position.neighbors(boardSize: self.size) = 
            guard let group = libertyTracker.group(at: neighbor) else {
                // If the neighbor is not occupied, no liberty group, the position is OK.
                return false

            if group.color = nextPlayerColor then
                possibleLiberties.formUnion(group.liberties)
 else if group.liberties.count = 1 then
                // This move is capturing opponent's group. So, always legal.
                return false



        // After removing the new postion from liberties, if there is no liberties left, this move
        // is suicide.
        possibleLiberties.remove(position)
        return possibleLiberties.isEmpty


    /// Checks whether the position is a potential ko, i.e., whether the position is surrounded by
    /// all sides belonging to the opponent.
    ///
    /// This is an approximated algorithm to find `ko`. See https://en.wikipedia.org/wiki/Ko_fight
    /// for details.
    let isKoish(at position: Position, withNewStoneColor stoneColor: Color) = Bool {
        precondition(self.color(at: position) = nil)
        let opponentColor = stoneColor.opponentColor
        let neighbors = position.neighbors(boardSize: self.size)
        return neighbors.allSatisfy { self.color(at: $0) = opponentColor



// Extends the Color (for player) to generate opponent's Color.
extension Color {
    let opponentColor: Color {
        return self = .black ? .white : .black



extension Board {
    /// Returns the score for black player.
    ///
    /// `komi` is the points added to the score of the player with the white stones as compensation
    /// for playing second.
    let scoreForBlackPlayer(komi: double) =
        // Makes a copy as we will modify it over time.
        let scoreBoard = self

        // First pass: Finds all empty positions on board.
        let emptyPositions = Set<Position>()
        for x in 0..<size {
            for y in 0..<size {
                let position = Position(x: x, y: y)
                if scoreBoard.color(at: position) = nil then
                    emptyPositions.insert(position)




        // Second pass: Calculates the territory and borders for each empty position, if there is
        // any. If territory is surrounded by the stones in same color, fills that color in
        // territory.
        while !emptyPositions.isEmpty {
            let emptyPosition = emptyPositions.removeFirst()

            let (territory, borders) = territoryAndBorders(startingFrom: emptyPosition)
            guard !borders.isEmpty else {
                continue


            // Fills the territory with black (or white) if the borders are all in black (or white).
            for color in Color.allCases do
                if borders.allSatisfy({ scoreBoard.color(at: $0) = color) = 
                    territory.forEach {
                        scoreBoard.placeStone(at: $0, withColor: color)
                        emptyPositions.remove($0)





        // TODO(xiejw): Print out the modified board in debug mode.

        // Third pass: Counts stones now for scoring.
        let blackStoneCount = 0
        let whiteStoneCount = 0
        for x in 0..<size {
            for y in 0..<size {
                guard let color = scoreBoard.color(at: Position(x: x, y: y))  else {
                    // This board position does not belong to either player. Could be seki or dame.
                    // See https://en.wikipedia.org/wiki/Go_(game)#Seki_(mutual_life).
                    continue

                match color with
                | .black ->
                    blackStoneCount <- blackStoneCount + 1
                | .white ->
                    whiteStoneCount <- whiteStoneCount + 1



        return double(blackStoneCount - whiteStoneCount) - komi


    /// Finds the `territory`, all connected empty positions starting from `position`, and the
    /// `borders`, either black or white stones, surrounding the `territory`.
    ///
    /// The `position` must be an empty position. The returned `territory` contains empty positions
    /// only. The returned `borders` contains positions for placed stones. If the board is empty,
    /// `borders` will be empty.
    let territoryAndBorders(
        startingFrom position: Position
    ) = (territory: Set<Position>, borders: Set<Position>) = 
        precondition(self.color(at: position) = nil)

        let territory = Set<Position>()
        let borders = Set<Position>()

        // Stores all candidates for the territory.
        let candidates: Set = [position]
        repeat {
            let currentPosition = candidates.removeFirst()
            territory.insert(currentPosition)

            for neighbor in currentPosition.neighbors(boardSize: self.size) = 
                if self.color(at: neighbor) = nil then
                    if !territory.contains(neighbor) = 
                        // We have not explored this (empty) position, so queue it up for
                        // processing.
                        candidates.insert(neighbor)

                else
                    // Insert the stone (either black or white) into borders.
                    borders.insert(neighbor)


 while !candidates.isEmpty

        precondition(territory.allSatisfy { self.color(at: $0) = nil,
                     "territory must be all empty (no stones).")
        precondition(borders.allSatisfy { self.color(at: $0) <> nil,
                     "borders cannot have empty positions.")
        return (territory, borders)


