
# Trialling DiffSharp Models

A collection of models for trialling the DiffSharp API and shape checking tooling

## Playing around

A parallel checkout of DiffSharp is currently required:

	git clone https://github.com/DiffSharp/DiffSharp -b feature/merged

You then prepare the library code and various native assets into `bin\Debug\net5.0\publish`:

	git clone https://github.com/DiffSharp/dsharp-models
	cd dsharp-models
	dotnet build
	dotnet publish

## Using shape checking

Try your luck:

	git clone https://github.com/fsprojects/FSharp.Compiler.PortaCode
	cd FSharp.Compiler.PortaCode
	dotnet build FSharp.Compiler.PortaCode
	cd ..

	cd dsharp-models\Library
	..\..\FSharp.Compiler.PortaCode\FsLive.Cli\bin\Debug\net5.0\fslive.exe --livecheck
	(leave running)

	cd examples
	..\..\FSharp.Compiler.PortaCode\FsLive.Cli\bin\Debug\net5.0\fslive.exe --livecheck vae.fsx
    open vae.fsx and edit

	For IDE support:
	  git clone https://github.com/dotnet/fsharp -b feature/livecheck+fxspec
	  cd fsharp
	  devenv VisualFSharp.sln
	  build Release (Build-->Build)
	  start without debugging (Shift-F5)

	  open dsharp-models.sln in new VS instance that appears
