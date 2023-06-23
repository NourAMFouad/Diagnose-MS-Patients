import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;

late tfl.Interpreter interpreter;
late String result;
List<double> outputList = [];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    interpreter = await tfl.Interpreter.fromAsset('model_unquant.tflite');
    print('Interpreting model done successfully ✔');
  } catch (e) {
    print('Error while interpreting model !!');
  }
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key}) : super(key: key);

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;

  late Object output;

  Future<void> _pickImage(ImageSource source) async {
    final image = await ImagePicker().getImage(source: source);
    if (image == null) return;
    // to show the result
    result = await predictImage(File(image.path));
    outputList = convertOutputToList(result);
    //assignOutputToVariables(outputList);
    useOutputList(outputList);

    print(result);

    setState(() {
      _image = File(image.path);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Final Result'),
        backgroundColor: const Color.fromARGB(248, 7, 62, 125),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            if (_image != null) ...[
              Container(
                height: 200, // set the height and width to the same value
                width: 200,
                child: Image.file(_image!),
              ),
              const SizedBox(height: 20),
              Text(
                //outputList.join(", "),
                useOutputList(outputList),
                style: const TextStyle(fontSize: 45),
              ),
              const SizedBox(height: 20),
            ],
            ElevatedButton(
              child: const Text('Pick Image'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color.fromARGB(248, 7, 62, 125),
              ),
              onPressed: () => _pickImage(ImageSource.gallery),
            ),
          ],
        ),
      ),
    );
  }

  Future<String> predictImage(File imageFile) async {
    try {
      // Load the image and resize it to the expected input size
      // decodeImage() function to decode image (convert image from bytes and return it as object)
      img.Image originalImage = img.decodeImage(await imageFile.readAsBytes())!;
      // get the size of input in the loaded TFlite model
      int inputSize = interpreter.getInputTensor(0).shape[1];
      // change the size of input image to use it in Tflite model
      img.Image resizedImage =
          img.copyResize(originalImage, width: inputSize, height: inputSize);

      // Convert the image to grayscale
      // to make the input image in tflite in grayScale
      img.Image grayscaleImage = img.grayscale(resizedImage);

      // Duplicate the grayscale channel to create a 3-channel image
      img.Image rgbImage = img.Image.from(grayscaleImage);

      // Normalize the image pixels (assuming pixel values in range [0, 255])
      var imageBytes = rgbImage.getBytes();
      var normalizedPixels =
          imageBytes.map((pixelValue) => pixelValue / 255.0).toList();

      // Create a 4-dimensional input tensor (assuming 3-channel image)
      var input = Float32List(inputSize * inputSize * 3)
          .reshape([1, inputSize, inputSize, 3]);
      for (int i = 0; i < inputSize * inputSize; ++i) {
        input[0][i ~/ inputSize][i % inputSize][0] = normalizedPixels[i * 3];
        input[0][i ~/ inputSize][i % inputSize][1] =
            normalizedPixels[i * 3 + 1];
        input[0][i ~/ inputSize][i % inputSize][2] =
            normalizedPixels[i * 3 + 2];
      }

      // Run the interpreter
      final inputShape = interpreter.getInputTensor(0).shape;
      if (inputShape[0] != 1) {
        throw Exception(
            'Invalid input batch size ${inputShape[0]}, expected 1.');
      }
      final outputShape = interpreter.getOutputTensor(0).shape;
      var output =
          Float32List(outputShape.reduce((a, b) => a * b)).reshape(outputShape);
      interpreter.run(input, output);

      return output.toString();
    } catch (e) {
      print('Error while predicting image: $e');
      return 'Error';
    }
  }

  List<double> convertOutputToList(String output) {
    List<String> outputStrings = output
        .replaceAll('[', '') // remove brackets
        .replaceAll(']', '')
        .split(', '); // split by comma and space
    return outputStrings.map((s) => double.parse(s)).toList();
  }

  String useOutputList(List<double> outputList) {
    double variable1 = outputList[0];
    double variable2 = outputList[1];
    // double variable3 = outputList[2];
    //double variable4 = outputList[3];

    if (variable1 > 0.25) {
      return 'MS patient';
    } else {
      return 'Not MS patient';
    }
  }
}
