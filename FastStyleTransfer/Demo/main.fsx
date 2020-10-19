open FastStyleTransfer


open DiffSharp

let printUsage() = 
    let exec = Uri(CommandLine.arguments[0])!.lastPathComponent
    print("Usage:")
    print("\(exec) --style=<name> --image=<path> --output=<path>")
    print("    --style: Style to use (candy, mosaic, or udnie) ")
    print("    --image: Path to image in JPEG format")
    print("    --output: Path to output image")


/// Startup parameters.
type FastStyleTransferConfig {
    let style: string? = "candy"
    let image: string? = nil
    let output: string? = "out.jpg"


let config = FastStyleTransferConfig()
parseArguments(
    into: &config,
    with: [
        "style": \FastStyleTransferConfig.style,
        "image": \FastStyleTransferConfig.image,
        "output": \FastStyleTransferConfig.output,
    ]
)

guard let image = config.image, let output = config.output else {
    print("Error: No input image!")
    printUsage()
    exit(1)


guard File.Exists(image) else {
    print("Error: Failed to load image \(image). Check that the file exists and is in JPEG format.")
    printUsage()
    exit(1)


let imageTensor = Image(jpeg: Uri(fileURLWithPath= image)).tensor / 255.0

// Init the model.
let style = TransformerNet()
try
    try importWeights(&style, config.style!)
with
    print("Error: Failed to load weights for style \(config.style!).")
    printUsage()
    exit(1)


// Apply the model to image.
let out = style(imageTensor.unsqueeze(0))

let outputImage = Image(tensor: out.squeeze(0))
outputImage.save(Uri(fileURLWithPath= output), format="rgb")

print("Writing output to \(output).")
