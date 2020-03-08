
window = {}
document = {}
importScripts("./onnx.min.js");

console.log(this)
const onnx = this.window.onnx
// create a session
const myOnnxSession = new onnx.InferenceSession();
// load the ONNX model file
const loadModelPromise = myOnnxSession.loadModel("cnndetection.onnx");

onmessage = async function(event) {
    console.log(event);
    var imgData = event.data

    // perform CenterCrop 224 ang get rgba data
    console.log(imgData)

    //https://stackoverflow.com/a/10475622/2523414
    //imgData is now an array where every 4 places are each pixel. So [0][1][2][3] are the [r][g][b][a] of the first pixel.
    
    // we normalize and build the final data ((x - x̄) / s)
    var norm_mean = [0.485, 0.456, 0.406]
    var norm_std = [0.229, 0.224, 0.225]


    var model_input = new Float32Array(3 * centerCropSize * centerCropSize)
    // for each color channel
    for (var ci = 0 ; ci < 3 ; ci ++) {
        for (var rowi = 0 ; rowi < centerCropSize ; rowi++) {
            for (var coli = 0 ; coli < centerCropSize ; coli++) {
                var idpi = (coli + rowi) * 4 /* input from canvas has an alpha channel */
                var tensor_val_i = imgData[idpi + ci] / 255.0 /* torchvision.transforms.ToTensor */
                var norm_tensor_val_i = (tensor_val_i - norm_mean[ci]) / norm_std[ci] /* torchvision.transforms.Normalize */
                model_input[(ci * centerCropSize * centerCropSize) + (rowi * centerCropSize) + coli] = norm_tensor_val_i
            }
        }
    }
    console.log(model_input)
    // note that we used a trick to avoid the onnx Shape operator
    // that prevent us from having a batch size greater than 1
    // more info: https://github.com/microsoft/onnxjs/issues/84#issuecomment-461682909
    const inputs = [
        new Tensor(model_input, "float32", [1, 3, centerCropSize, centerCropSize])
    ];
    console.log(inputs)
    await loadModelPromise;
    const outputMap = await myOnnxSession.run(inputs);
    const outputTensor = outputMap.values().next().value;
    const res = outputTensor.data[0]
    //const res = 11.138656 // fake
    //const res = -31.707857 // real
    const prob = 1/(1 + Math.exp(-res)) /* sigmoid */
    console.log({res})
    console.log({prob})

    postMessage(prob);
}