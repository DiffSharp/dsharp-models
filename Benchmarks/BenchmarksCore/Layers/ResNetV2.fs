
namespace Benchmark
(*
open ImageClassificationModels

let ResNetV2Suites = [
  makeLayerSuite(
    name= "ResNet18v2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) = 
    ResNetV2(classCount: 1000, depth: ResNet18)
  },
  makeLayerSuite(
    name= "ResNet34v2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) = 
    ResNetV2(classCount: 1000, depth: ResNet34)
  },
  makeLayerSuite(
    name= "ResNet50v2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) = 
    ResNetV2(classCount: 1000, depth: ResNet50)
  },
  makeLayerSuite(
    name= "ResNet101v2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) = 
    ResNetV2(classCount: 1000, depth: ResNet101)
  },
  makeLayerSuite(
    name= "ResNet152v2",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) = 
    ResNetV2(classCount: 1000, depth: ResNet152)
  },
]
*)
