
namespace TrainingLoop
(*
open DiffSharp

/// Metrics that can be registered into TrainingLoop.
public enum TrainingMetrics {
  case loss
  case accuracy

  let name: string {
    match self with
    | .loss ->
      return "loss"
    | .accuracy ->
      return "accuracy"
    }
  }

  let measurer: MetricsMeasurer {
    match self with
    | .loss ->
      return LossMeasurer(self.name)
    | .accuracy ->
      return AccuracyMeasurer(self.name)
    }
  }
}

/// An accumulator of statistics.
type IMetricsMeasurer {
  /// Name of the metrics.
  let name: string { get set }

  /// Clears accumulated data up and resets measurer to initial state.
  mutating let reset()

  /// Accumulates data from `loss`, `predictions`, `labels`.
  mutating let accumulate<Output, Target>(
    loss: Tensor?, predictions: Output?, labels: Target?
  )

  /// Computes metrics from cumulated data.
  let measure() = Float
}

/// A measurer for measuring loss.
type LossMeasurer: MetricsMeasurer {
  /// Name of the LossMeasurer.
  let name: string

  /// Sum of losses cumulated from batches.
  let totalBatchLoss: Float = 0

  /// Count of batchs cumulated so far.
  let batchCount: int32 = 0

  /// Creates an instance with the LossMeasurer named `name`.
  public init(_ name: string = "loss") = 
    self.name = name
  }

  /// Resets totalBatchLoss and batchCount to zero.
  public mutating let reset() = 
    totalBatchLoss = 0
    batchCount = 0
  }

  /// Adds `loss` to totalBatchLoss and increases batchCount by one.
  public mutating let accumulate<Output, Target>(
    loss: Tensor?, predictions: Output?, labels: Target?
  ) = 
    if let newBatchLoss = loss {
      totalBatchLoss <- totalBatchLoss + newBatchLoss.scalarized()
      batchCount <- batchCount + 1
    }
  }

  /// Computes averaged loss.
  let measure() =
    return totalBatchLoss / Float(batchCount)
  }
}

/// A measurer for measuring accuracy
type AccuracyMeasurer: MetricsMeasurer {
  /// Name of the AccuracyMeasurer.
  let name: string

  /// Count of correct guesses.
  let correctGuessCount: int32 = 0

  /// Count of total guesses.
  let totalGuessCount: int32 = 0

  /// Creates an instance with the AccuracyMeasurer named `name`. 
  public init(_ name: string = "accuracy") = 
    self.name = name
  }

  /// Resets correctGuessCount and totalGuessCount to zero.
  public mutating let reset() = 
    correctGuessCount = 0
    totalGuessCount = 0
  }

  /// Computes correct guess count from `loss`, `predictions` and `labels`
  /// and adds it to correctGuessCount; Computes total guess count from
  /// `labels` shape and adds it to totalGuessCount.
  public mutating let accumulate<Output, Target>(
    loss: Tensor?, predictions: Output?, labels: Target?
  ) = 
    guard let predictions = predictions as? Tensor<Float>, let labels = labels as? Tensor<int32>
    else {
      fatalError(
        "For accuracy measurements, the model output must be Tensor<Float>, and the labels must be Tensor<Int>."
      )
    }
    correctGuessCount <- correctGuessCount + Tensor<int32>(predictions.argmax(squeezingAxis: -1) .== labels).sum()
      .scalarized()
    totalGuessCount <- totalGuessCount + int32(labels.shape.reduce(1, * ))
  }

  /// Computes accuracy as percentage of correct guesses.
  let measure() =
    return Float(correctGuessCount) / Float(totalGuessCount)
  }
}
*)
