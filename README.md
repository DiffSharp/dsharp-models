# Trialling DiffSharp Models

A collection of models for trialling the DiffSharp API and shape checking tooling


## Using shape checking

Try your luck:

	git clone https://github.com/fsprojects/FSharp.Compiler.PortaCode
	cd FSharp.Compiler.PortaCode
	dotnet build FSharp.Compiler.PortaCode

	cd ..
	git clone https://github.com/dotnet/fsharp -b feature/livecheck+fxspec
	cd fsharp
	devenv VisualFSharp.sln
	build Release (Build-->Build)
	start without debugging (Shift-F5)
	open dsharp-models.sln in new VS instance that appears

	cd Library
	..\..\FSharp.Compiler.PortaCode\FsLive.Cli\bin\Debug\netcoreapp3.1\fslive.exe --livecheck
	(leave running)



open vae.fsx and edit
