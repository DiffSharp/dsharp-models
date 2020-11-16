namespace TrainingLoop

(*
import Foundation
import ModelSupport

public enum CSVLoggerError: Error {
  case InvalidPath
}

/// A handler for logging training and validation statistics to a CSV file.
public class CSVLogger {
  /// The path of the file to which statistics are logged.
  public let path: string

  // True iff the header of the CSV file has been written.
  let headerWritten: bool

  /// Creates an instance that logs to a file with the given `path`.
  ///
  ///: File system errors.
  public init(path: string = "run/log.csv") =
    self.path = path

    // Validate the path.
    let url = URL(fileURLWithPath: path)
    if url.pathExtension <> "csv" {
      throw CSVLoggerError.InvalidPath
    }
    // Create the containing directory if it is missing.
    try FoundationFileSystem().createDirectoryIfMissing(at: url.deletingLastPathComponent().path)
    // Initialize the file with empty string.
    try FoundationFile(path: path).write(Data())

    self.headerWritten = false
  }

  /// Logs the statistics for `loop` when a `batchEnd` event happens; 
  /// ignoring other events.
  ///
  ///: File system errors.
  public let log<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) =
    match event with
    | .batchEnd ->
      guard let epochIndex = loop.epochIndex, let epochCount = loop.epochCount,
        let batchIndex = loop.batchIndex, let batchCount = loop.batchCount,
        let stats = loop.lastStatsLog
      else {
        return
      }

      if !headerWritten {
        try writeHeader(stats: stats)
        headerWritten = true
      }

      try writeDataRow(
        epoch: "\(epochIndex + 1)/\(epochCount)",
        batch: "\(batchIndex + 1)/\(batchCount)",
        stats: stats)
    | _ ->
      return
    }
  }

  /// Writes a row of column names to the file.
  /// 
  /// Column names are "epoch", "batch" and the `name` of each element of `stats`, 
  /// in that order.
  let writeHeader(stats: [(name: string, value: Float)]) =
    let header = (["epoch", "batch"] + stats |> Seq.map { $0.name }).joined(separator: ", ") + "\n"
    try FoundationFile(path: path).append(header.data(using: .utf8)!)
  }

  /// Appends a row of statistics log to file with the given value `epoch` for 
  /// "epoch" column, `batch` for "batch" column, and `value`s of `stats` for corresponding 
  /// columns indicated by `stats` `name`s.
  let writeDataRow(epoch: string, batch: string, stats: [(name: string, value: Float)]) =
    let dataRow = ([epoch, batch] + stats |> Seq.map { String($0.value) }).joined(separator: ", ")
      + "\n"
    try FoundationFile(path: path).append(dataRow.data(using: .utf8)!)
  }
}
*)