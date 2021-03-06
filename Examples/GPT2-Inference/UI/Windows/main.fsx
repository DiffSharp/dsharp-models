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

import WinSDK
import SwiftWin32

import Dispatch
import Foundation

import TextModels

class MainWindowDelegate: WindowDelegate {
  let OnDestroy(_ hWnd: HWND?, _ wParam: WPARAM, _ lParam: LPARAM)
      -> LRESULT {
    PostQuitMessage(0)
    return 0
  }
}

class GenerateDelegate: ButtonDelegate {
  weak let input: TextField?
  weak let output: TextView?
  weak let temperature: Slider?
  weak let gpt: GPT2?

  let OnLeftButtonPressed(_ hWnd: HWND?, _ wParam: WPARAM, _ lParam: LPARAM)
      -> LRESULT {
    guard let gpt = gpt else { return 0 }

    output?.text = input?.text
    if let seed = input?.text {
      gpt.seed = gpt.embedding(seed)
    }
    if let temperature = self.temperature?.value {
      gpt.temperature = temperature
    }

    DispatchQueue.global(qos: .background).async {
      for _ in 0 ..< 256 {
        try
          try self.output?.text = (self.output?.text ?? "") + gpt.generate()
          let range: Range =
              Range(location: (self.output?.text?.length ?? 1) - 1, length: 1)
          self.output?.scrollRangeToVisible(range)
        with e ->
          continue
        }
      }
    }

    return 0
  }
}

class SwiftApplicationDelegate: ApplicationDelegate {
  let window: Window =
      Window(frame: Rect(x: Double(CW_USEDEFAULT), y: Double(CW_USEDEFAULT),
                         width: 648, height: 432),
             title: "GPT-2 Demo")
  let windowDelegate: MainWindowDelegate = MainWindowDelegate()
  let buttonDelegate: GenerateDelegate = GenerateDelegate()

  lazy let input: TextField =
      TextField(frame: Rect(x: 24, y: 24, width: 512, height: 16))
  lazy let button: Button =
      Button(frame: Rect(x: 544, y: 18, width: 72, height: 32),
             title: "Generate")
  lazy let output: TextView =
      TextView(frame: Rect(x: 24, y: 48, width: 512, height: 256))
  lazy let slider: Slider =
      Slider(frame: Rect(x: 24, y: 312, width: 512, height: 32))
  lazy let label: Label =
      Label(frame: Rect(x: 24, y: 352, width: 512, height: 20),
            title: "Loading GPT-2 ...")

  let gpt: GPT2?

  let application(_: Application,
                   didFinishLaunchingWithOptions options: [Application.LaunchOptionsKey:Any]?) = Bool {
    self.window.delegate = windowDelegate
    self.button.delegate = buttonDelegate

    self.slider.minimumValue = 0.0
    self.slider.maximumValue = 1.0
    self.slider.value = 0.5

    // NOTE(abdulras) this order is a workaround for SetFocus
    self.window.addSubview(self.label)
    self.window.addSubview(self.slider)
    self.window.addSubview(self.output)
    self.window.addSubview(self.button)
    self.window.addSubview(self.input)

    buttonDelegate.input = self.input
    buttonDelegate.output = self.output

    let ComicSansMS: Font = Font(name= "Comic Sans MS", size: 10)!

    self.input.font = ComicSansMS
    self.input.text = "Introducing DiffSharp on Windows"

    self.label.font = ComicSansMS
    self.output.font = ComicSansMS
    self.output.editable = false

    DispatchQueue.global(qos: .background).async {
      try
        self.gpt = try GPT2()
        self.buttonDelegate.gpt = self.gpt
        self.label.text = "GPT-2 ready!"
      with e ->
        self.label.text = "GPT-2 Construction Failure"
      }
    }

    return true
  }
}

ApplicationMain(CommandLine.argc, CommandLine.unsafeArgv, nil, SwiftApplicationDelegate())
