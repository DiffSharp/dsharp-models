
# Trialling DiffSharp Models

A collection of models for trialling the DiffSharp API and shape checking tooling

## Playing around

A parallel checkout of dotnet/fsharp and DiffSharp/DiffSharp are currently required:

        git clone https://github.com/dotnet/fsharp -b feature/analyzers
	cd fsharp
	.\build -pack -deploy
	cd ..

        git clone https://github.com/fsprojects/FSharp.Compiler.PortaCode -b feature/analyzers
	cd FSharp.Compiler.PortaCode
	dotnet build FSharp.Compiler.PortaCode
	cd ..
	
        git clone https://github.com/DiffSharp/DiffSharp -b feature/merged

	git clone https://github.com/DiffSharp/dsharp-models
	dotnet build
	# You then prepare the library code and various native assets into `bin\Debug\net5.0\publish`:
        dotnet publish

	devenv dsharp-models.sln /RootSuffix RoslynDev



