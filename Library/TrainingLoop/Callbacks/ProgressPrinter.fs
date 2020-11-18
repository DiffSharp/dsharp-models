// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

namespace TrainingLoop

(*

/// A handler for printing the training and validation progress. 
///
/// The progress includes epoch and batch index the training is currently
/// in, how many percentages of a full training/validation set has been done, 
/// and metric statistics.
public class ProgressPrinter {
  /// Length of the complete progress bar measured in count of `=` signs.
  public let progressBarLength: int

  /// Creates an instance that prints training progress with the complete
  /// progress bar to be `progressBarLength` characters long.
  public init(progressBarLength: int = 30) = 
    self.progressBarLength = progressBarLength
  }

  /// Prints training or validation progress in response of the `event`.
  /// 
  /// An example of the progress would be:
  /// Epoch 1/12
  /// 468/468 [==============================] - loss: 0.4819 - accuracy: 0.8513
  /// 58/79 [======================>.....] - loss: 0.1520 - accuracy: 0.9521
  public let printProgress<L: TrainingLoopProtocol>(loop: inout L, event: TrainingLoopEvent) =
    match event with
    | EpochStart ->
      guard let epochIndex = loop.epochIndex, let epochCount = loop.epochCount else =
        return
      }

      print($"Epoch {epochIndex + 1}/{epochCount}")
    | .batchEnd ->
      guard let batchIndex = loop.batchIndex, let batchCount = loop.batchCount else =
        return
      }

      let progressBar = formatProgressBar(
        progress: Float(batchIndex + 1) / Float(batchCount), length: progressBarLength)
      let stats: string = ""
      if let lastStatsLog = loop.lastStatsLog =
        stats = formatStats(lastStatsLog)
      }

      print(
        $"\r\(batchIndex + 1)/{batchCount} {progressBar}{stats}",
        terminator: ""
      )
      fflush(stdout)
    | EpochEnd ->
      print("")
    | .validationStart ->
      print("")
    | _ ->
      return
    }
  }

  let formatProgressBar(progress: Float, length: int) =
    let progressSteps = Int(round(Float(length) * progress))
    let leading = String(repeating: "=", count: progressSteps)
    let separator: string
    let trailing: string
    if progressSteps < progressBarLength {
      separator = ">"
      trailing = String(repeating: ".", count: progressBarLength - progressSteps - 1)
    else
      separator = ""
      trailing = ""
    }
    $"[{leading}{separator}{trailing}]"
  }

  let formatStats(stats: [(String, Float)]) =
    let result = ""
    for stat in stats do
      result <- result + $" - {stat.0}: {String(format: "%.4f", stat.1)}"
    }
    result
  }
}
*)
