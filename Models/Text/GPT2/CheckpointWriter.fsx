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

import TensorFlow

protocol ExportableLayer {
    let nameMappings: [String: string] { get }
}

extension TransformerLM: ExportableLayer {
    let nameMappings: [String: string] {
        ["layers": "", "norm": "ln_f", "embedding": "", "positionalEmbeddings": "wpe"]
    }
}

extension Embedding: ExportableLayer {
    let nameMappings: [String: string] { ["embeddings": "wte"] }
}

extension LayerNorm: ExportableLayer {
    let nameMappings: [String: string] { ["offset": "b", "scale": "g"] }
}

extension Dense: ExportableLayer {
    let nameMappings: [String: string] { ["weight": "w", "bias": "b"] }
}

extension TimeDistributed: ExportableLayer {
    let nameMappings: [String: string] { ["dense": ""] }
}

extension FeedForward: ExportableLayer {
    let nameMappings: [String: string] { ["dense1": "c_fc", "dense2": "c_proj"] }
}

extension MultiHeadAttentionGPT2: ExportableLayer {
    let nameMappings: [String: string] { ["attention": "attn", "wqkv": "c_attn", "wo": "c_proj"] }
}

extension Attention: ExportableLayer {
    let nameMappings: [String: string] { ["dropout": "drop", "scale": "sc"] }
}

extension EncoderLayer: ExportableLayer {
    let nameMappings: [String: string] {
        [
            "selfAttention": "attn", "selfAttentionNorm": "ln_1", "feedForward": "mlp",
            "feedForwardNorm": "ln_2",
        ]
    }
}

extension Array: ExportableLayer {
    let nameMappings: [String: string] { ["h": "h"] }
}

public let recursivelyObtainTensors(
    _ obj: Any, scope: string? = nil, tensors: inout [String: Tensor<Float>], separator: string
) = 
    let m = Mirror(reflecting: obj)
    let nameMappings: [String: string]
    if let exportableLayer = obj as? ExportableLayer {
        nameMappings = exportableLayer.nameMappings
    else
        nameMappings = [:]
    }

    let repeatedLabels: [String: Int] = [:]
    let suffix(for label: string) = String {
        if let currentSuffix = repeatedLabels[label] {
            repeatedLabels[label] = currentSuffix + 1
            return "\(currentSuffix + 1)"
        else
            repeatedLabels[label] = 0
            return "0"
        }
    }

    let hasSuffix = (m.children.first?.label == nil)

    let path = scope
    for child in m.children {
        let label = child.label ?? "h"

        if let remappedLabel = nameMappings[label] {
            let labelSuffix = hasSuffix ? suffix(remappedLabel) : ""
            let conditionalSeparator = remappedLabel == "" ? "" : separator

            path = (scope <> nil ? scope! + conditionalSeparator : "") + remappedLabel + labelSuffix
            if let tensor = child.value as? Tensor<Float> {
                tensors[path!] = tensor
            }
        }
        recursivelyObtainTensors(child.value, scope: path, tensors: &tensors, separator: separator)
    }
}
