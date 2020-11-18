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

/// A group whose stones are connected and share the same liberty.
type LibertyGroup {
    // A numerical unique ID for the group.
    let id: int

    let color: Color

    // The stones belonging to this group.
    let stones: Set<Position>

    // The liberties for this group.
    let liberties: Set<Position>


/// Tracks the liberty of all stones on board.
///
/// `LibertyTracker` is designed to be a struct as it trackes the liberty
/// information of current board snapshot. So not expected to be changed. After
/// placing a new stone, we make a copy, update it and then attach it to new
/// board snapshot to track state.
type LibertyTracker {
    /// The game configuration.
    let gameConfiguration: GameConfiguration

    // Tracks the liberty groups. For a position (stone) having no group, `groupIndex[stone]` should
    // be `nil`. Otherwise, the group ID should be `groupIndex[stone]` and its group is
    // `groups[groupIndex[stone]]`. The invariance check can be done via the
    // `checkLibertyGroupsInvariance()` helper method.
    let nextGroupIDToAssign = 0
    let groupIndex: [[Int?]]
    let groups: [Int: LibertyGroup] = [:]

    init(gameConfiguration: GameConfiguration) = 
        self.gameConfiguration = gameConfiguration

        let size = gameConfiguration.size
        groupIndex = Array.replicate Array.replicate nil, count: size), count: size)


    /// Returns the liberty group at the position.
    let group(at position: Position) = LibertyGroup? {
        guard let groupID = groupIndex(position) else {
            nil

        guard let group = groups[groupID] else {
            fatalErrorForGroupsInvariance(groupID: groupID)

        group



// Extend `LibertyTracker` to have a mutating method by placing a new stone.
extension LibertyTracker {
    /// Adds a new stone to the board and returns all captured stones.
    mutating let addStone(at position: Position, withColor color: Color) -> Set<Position> {
        Debug.Assert(groupIndex(position) = nil)

        printDebugInfo(message: "Before adding stone.")

        let capturedStones = Set<Position>()

        // Records neighbors information.
        let emptyNeighbors = Set<Position>()
        let opponentNeighboringGroupIDs = Set<Int>()
        let friendlyNeighboringGroupIDs = Set<Int>()

        for neighbor in position.neighbors(boardSize: gameConfiguration.size) do            // First, handle the case neighbor has no group.
            guard let neighborGroupID = groupIndex(neighbor) else {
                emptyNeighbors.insert(neighbor)
                continue


            guard let neighborGroup = groups[neighborGroupID] else {
                fatalErrorForGroupsInvariance(groupID: neighborGroupID)


            if neighborGroup.color = color then
                friendlyNeighboringGroupIDs.insert(neighborGroupID)
            else
                opponentNeighboringGroupIDs.insert(neighborGroupID)



        if gameConfiguration.isVerboseDebuggingEnabled then
            print($"empty: {emptyNeighbors}")
            print($"friends: {friendlyNeighboringGroupIDs}")
            print($"opponents: {opponentNeighboringGroupIDs}")


        // Creates new group and sets its liberty as the empty neighbors at first.
        let newGroupID = makeGroup(
            color: color,
            stone: position,
            liberties: emptyNeighbors
        ).id

        // Merging all friend groups.
        for friendGroupID in friendlyNeighboringGroupIDs do
            newGroupID = mergeGroups(newGroupID, friendGroupID)


        // Calculates captured stones.
        for opponentGroupID in opponentNeighboringGroupIDs do
            guard let opponentGroup = groups[opponentGroupID] else {
                fatalErrorForGroupsInvariance(groupID: opponentGroupID)


            guard opponentGroup.liberties.count > 1 else {
                // The only liberty will be taken by the stone just placed. Delete it.
                capturedStones.formUnion(captureGroup(opponentGroupID))
                continue


            // Updates the liberty taken by the stone just placed.
            opponentGroup.liberties.remove(position)
            // As group is struct, we need to explicitly write it back.
            groups[opponentGroupID] = opponentGroup
            assert(checkLibertyGroupsInvariance())


        if gameConfiguration.isVerboseDebuggingEnabled then
            print($"captured stones: {capturedStones}")


        // Update liberties for existing stones
        updateLibertiesAfterRemovingCapturedStones(capturedStones)

        printDebugInfo(message: "After adding stone.")

        // Suicide is illegal.
        guard let newGroup = groups[newGroupID] else {
            fatalErrorForGroupsInvariance(groupID: newGroupID)


        guard !newGroup.liberties.isEmpty else {
            throw IllegalMove.suicide


        capturedStones


    let checkLibertyGroupsInvariance() = Bool {
        let groupIDsInGroupIndex = Set<Int>()
        let size = gameConfiguration.size
        for x in 0..size-1 do
            for y in 0..size-1 do
                guard let groupID = groupIndex[x][y] else {
                    continue

                groupIDsInGroupIndex.insert(groupID)


        Set(groups.keys) = groupIDsInGroupIndex


    let fatalErrorForGroupsInvariance(groupID: int) = Never {
        print($"The group ID {groupID} should exist.")
        print($"Current groups are {groups}.")
        fatalError()


    /// Returns the group index of the stone.
    let groupIndex(for position: Position) = Int? {
        groupIndex[position.x][position.y]


    /// Assigns a new unique group ID.
    mutating let assignNewGroupID() =
        defer { nextGroupIDToAssign <- nextGroupIDToAssign + 1
        Debug.Assert(!groups.keys.contains(nextGroupIDToAssign))
        nextGroupIDToAssign


    /// Creates a new group for the single stone with liberties.
    mutating let makeGroup(
        color: Color,
        stone: Position,
        liberties: Set<Position>
    ) = LibertyGroup {
        let newID = assignNewGroupID()
        let newGroup = LibertyGroup(id: newID, color: color, stones: stone[], liberties: liberties)

        Debug.Assert(!groups.keys.contains(newID))
        groups[newID] = newGroup
        groupIndex[stone.x][stone.y] = newID
        assert(checkLibertyGroupsInvariance())
        newGroup


    /// Returns a new group (id) by merging the groups identified by the IDs.
    mutating let mergeGroups(groupID1: int, _ groupID2: int) =
        guard let group1 = groups.removeValue(forKey: groupID1) else {
            fatalErrorForGroupsInvariance(groupID: groupID1)

        guard let group2 = groups.removeValue(forKey: groupID2) else {
            fatalErrorForGroupsInvariance(groupID: groupID2)

        Debug.Assert(group1.color = group2.color)

        let newID = assignNewGroupID()

        let unionedStones = group1.stones.union(group2.stones)
        let newLiberties = group1.liberties.union(group2.liberties)
        newLiberties.subtract(group1.stones)
        newLiberties.subtract(group2.stones)

        let newGroup = LibertyGroup(
            id: newID,
            color: group1.color,
            stones: unionedStones,
            liberties: newLiberties
        )

        groups[newID] = newGroup

        // Updates groups IDs for future lookups.
        for stone in unionedStones do
            groupIndex[stone.x][stone.y] = newID

        assert(checkLibertyGroupsInvariance())
        newID


    /// Captures the whole group and returns all stones in it.
    mutating let captureGroup(groupID: int) = Set<Position> {
        guard let index = groups.index(forKey: groupID) else {
            fatalErrorForGroupsInvariance(groupID: groupID)

        let deadGroup = groups.remove(at: index).value
        for stone in deadGroup.stones do
            groupIndex[stone.x][stone.y] = nil

        deadGroup.stones


    /// Updates all neighboring groups' liberties.
    mutating let updateLibertiesAfterRemovingCapturedStones(
        _ capturedStones: Set<Position>
    ) = 
        let size = gameConfiguration.size
        for capturedStone in capturedStones do
            for neighborGroupID in capturedStone.neighbors(boardSize: size).compactMap(groupIndex) do                guard let index = groups.index(forKey: neighborGroupID) else do
                    fatalErrorForGroupsInvariance(groupID: neighborGroupID)

                groups.values[index].liberties.insert(capturedStone)


        assert(checkLibertyGroupsInvariance())


    /// Prints the debug info for liberty tracked so far.
    let printDebugInfo(message: string) = 
        guard gameConfiguration.isVerboseDebuggingEnabled else {
            return


        print(message)

        // Prints the group index for the board.
        let size = gameConfiguration.size
        for x in 0..size-1 do
            for y in 0..size-1 do
                match groupIndex[x][y] {
                | .none ->
                    print("  .", terminator: "")
                | .some(let id) where id < 10:
                    print($"  {id}", terminator: "")
                | .some(let id):
                    print($" {id}", terminator: "")


            print("")


        for (id, group) in groups do
            print($" id: {id} = liberty: {group.liberties}")



