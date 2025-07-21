import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(CameraAIApp());
}

class CameraAIApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: CameraHomePage(),
    );
  }
}

class CameraHomePage extends StatefulWidget {
  @override
  _CameraHomePageState createState() => _CameraHomePageState();
}

class _CameraHomePageState extends State<CameraHomePage> {
  final List<String> cocoClasses = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'TV',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
  ];

  bool _cameraPaused = false;
  CameraController? controller;
  String selectedModel = 'Model A';
  final List<String> modelOptions = ['Model A', 'Model B', 'Model C'];
  Interpreter? interpreter;
  bool modelLoaded = false;
  String detectedLabel = '[none]';
  List<Rect> boxes = [];

  DateTime _lastProcessed = DateTime.now().subtract(Duration(seconds: 5));

  @override
  void initState() {
    super.initState();
    initCameraAndModel();
  }

  Future<void> initCameraAndModel() async {
    if (cameras.isNotEmpty) {
      controller = CameraController(cameras[0], ResolutionPreset.medium);
      await controller!.initialize();
      controller!.startImageStream(onLatestImageAvailable);
      if (!mounted) return;
      setState(() {});
    }

    try {
      interpreter = await Interpreter.fromAsset(
        'assets/models/yolov5s-fp16.tflite',
      );
      print('✅ YOLOv5 model loaded.');
      setState(() => modelLoaded = true);
    } catch (e) {
      print('❌ Failed to load YOLOv5 model: $e');
    }
  }

  void _toggleCamera() async {
    if (_cameraPaused) {
      // Resume: reinitialize camera
      controller = CameraController(cameras[0], ResolutionPreset.medium);
      await controller!.initialize();
      controller!.startImageStream(onLatestImageAvailable);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('📸 Camera resumed'),
          duration: Duration(milliseconds: 800),
        ),
      );
    } else {
      // Pause: stop stream and dispose controller
      await controller?.stopImageStream();
      await controller?.dispose();
      controller = null;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('⏸️ Camera paused'),
          duration: Duration(milliseconds: 800),
        ),
      );
    }
    setState(() => _cameraPaused = !_cameraPaused);
  }

  bool _isProcessing = false;

  void onLatestImageAvailable(CameraImage image) async {
    if (_isProcessing || !modelLoaded || interpreter == null) return;

    // 🛑 Only process once every 5 seconds
    final now = DateTime.now();
    if (now.difference(_lastProcessed).inSeconds < 5) return;
    _lastProcessed = now;

    _isProcessing = true;

    try {
      const int inputSize = 640;
      final input = await preprocessCameraImage(image, inputSize);

      final inputTensor = [input];
      final outputBuffer = List.generate(
        1,
        (_) => List.generate(25200, (_) => List.filled(85, 0.0)),
      );

      interpreter!.runForMultipleInputs(inputTensor, {0: outputBuffer});

      final rawResults = outputBuffer[0];
      final List<Map<String, dynamic>> detections = [];

      for (var row in rawResults) {
        final conf = row[4];
        if (conf > 0.4) {
          final classScores = row.sublist(5);
          final maxScore = classScores.reduce((a, b) => a > b ? a : b);
          final classIndex = classScores.indexOf(maxScore);

          if (maxScore > 0.25) {
            detections.add({
              'box': row.sublist(0, 4),
              'confidence': conf,
              'class': classIndex,
              'score': maxScore,
            });
          }
        }
      }

      // After your detections are collected
      if (detections.isNotEmpty) {
        detections.sort(
          (a, b) =>
              (b['confidence'] as double).compareTo(a['confidence'] as double),
        );
        final best = detections.first;
        final classIndex = best['class'] as int;

        if (mounted) {
          setState(() {
            detectedLabel = cocoClasses[classIndex];
            // Clear old boxes
            boxes.clear();
            // Extract bbox from detection
            final box = best['box'] as List<double>;
            // box is [centerX, centerY, width, height] normalized, convert to Rect with top-left and bottom-right
            final double centerX = box[0];
            final double centerY = box[1];
            final double w = box[2];
            final double h = box[3];

            // Calculate top-left
            final left = centerX - w / 2;
            final top = centerY - h / 2;

            // Add rect normalized, you will scale this in painter
            boxes.add(Rect.fromLTWH(left, top, w, h));
          });
        }
      } else {
        if (mounted) {
          setState(() {
            detectedLabel = '[none]';
            boxes.clear();
          });
        }
      }
    } catch (e) {
      print('Error in image processing: $e');
    } finally {
      _isProcessing = false;
    }
  }

  Future<List<List<List<List<double>>>>> preprocessCameraImage(
    CameraImage image,
    int inputSize,
  ) async {
    final img.Image rgbImage = convertYUV420ToImage(image);
    final img.Image resized = img.copyResize(
      rgbImage,
      width: inputSize,
      height: inputSize,
    );

    final result = List.generate(
      1,
      (_) => List.generate(
        inputSize,
        (y) => List.generate(inputSize, (x) {
          final pixel = resized.getPixel(x, y);
          final r = pixel.r / 255.0;
          final g = pixel.g / 255.0;
          final b = pixel.b / 255.0;
          return [r, g, b];
        }),
      ),
    );

    return result;
  }

  /// Converts [CameraImage] in YUV420 format to an RGB [img.Image] from the `image` package.
  img.Image convertYUV420ToImage(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    final img.Image rgbImage = img.Image(width: width, height: height);

    final Uint8List yPlane = image.planes[0].bytes;
    final Uint8List uPlane = image.planes[1].bytes;
    final Uint8List vPlane = image.planes[2].bytes;

    final int yRowStride = image.planes[0].bytesPerRow;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int yIndex = y * yRowStride + x;

        final int uvX = x ~/ 2;
        final int uvY = y ~/ 2;
        final int uvIndex = uvY * uvRowStride + uvX * uvPixelStride;

        final int yValue = yPlane[yIndex] & 0xFF;
        final int uValue = uPlane[uvIndex] & 0xFF;
        final int vValue = vPlane[uvIndex] & 0xFF;

        final int r =
            (yValue + 1.370705 * (vValue - 128)).clamp(0, 255).toInt();
        final int g =
            (yValue - 0.337633 * (uValue - 128) - 0.698001 * (vValue - 128))
                .clamp(0, 255)
                .toInt();
        final int b =
            (yValue + 1.732446 * (uValue - 128)).clamp(0, 255).toInt();

        rgbImage.setPixelRgb(x, y, r, g, b);
      }
    }

    return rgbImage;
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Camera or placeholder
          Positioned.fill(
            child:
                !_cameraPaused &&
                        controller != null &&
                        controller!.value.isInitialized
                    ? CameraPreview(controller!)
                    : Container(
                      color: Colors.black,
                      alignment: Alignment.center,
                      child: Text(
                        '📷 Camera Paused',
                        style: TextStyle(color: Colors.white70, fontSize: 20),
                      ),
                    ),
          ),
          // Detected label
          Align(
            alignment: Alignment.bottomLeft,
            child: Container(
              margin: EdgeInsets.all(16),
              padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                'Detected: $detectedLabel',
                style: TextStyle(color: Colors.white, fontSize: 16),
              ),
            ),
          ),
          // Model selector
          Align(
            alignment: Alignment.bottomRight,
            child: Container(
              margin: EdgeInsets.all(16),
              padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(12),
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  dropdownColor: Colors.black,
                  value: selectedModel,
                  style: TextStyle(color: Colors.white),
                  items:
                      modelOptions
                          .map(
                            (model) => DropdownMenuItem(
                              value: model,
                              child: Text(model),
                            ),
                          )
                          .toList(),
                  onChanged: (val) {
                    setState(() => selectedModel = val!);
                  },
                ),
              ),
            ),
          ),
          // Pause/Resume button
          Align(
            alignment: Alignment.topRight,
            child: Container(
              margin: EdgeInsets.all(16),
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.black87,
                  padding: EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                onPressed: _toggleCamera,
                child: Text(
                  _cameraPaused ? 'Resume' : 'Pause',
                  style: TextStyle(color: Colors.white),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }


}
class BoundingBoxPainter extends CustomPainter {
  final List<Rect> boxes;
  final Color color;

  BoundingBoxPainter({required this.boxes, this.color = Colors.red});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    for (final box in boxes) {
      // box is normalized, convert to screen pixels
      final rect = Rect.fromLTWH(
        box.left * size.width,
        box.top * size.height,
        box.width * size.width,
        box.height * size.height,
      );
      canvas.drawRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(covariant BoundingBoxPainter oldDelegate) {
    return oldDelegate.boxes != boxes;
  }
}
