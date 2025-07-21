import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';
import 'dart:math';

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
  CameraController? controller;
  String selectedModel = 'Model A';
  final List<String> modelOptions = ['Model A', 'Model B', 'Model C'];
  Interpreter? interpreter;
  bool modelLoaded = false;
  String detectedLabel = '[none]';
  List<Rect> boxes = [];

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
      interpreter = await Interpreter.fromAsset('assets/models/yolov5s-fp16.tflite');
      print('✅ YOLOv5 model loaded.');
      setState(() => modelLoaded = true);
    } catch (e) {
      print('❌ Failed to load YOLOv5 model: $e');
    }
  }

  void onLatestImageAvailable(CameraImage image) async {
    if (!modelLoaded) return;

    // Simulate detection output for demo purposes
    setState(() {
      detectedLabel = 'Example Object';
      boxes = [
        Rect.fromLTWH(50, 100, 120, 200),
        Rect.fromLTWH(200, 300, 100, 150)
      ];
    });
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (controller == null || !controller!.value.isInitialized) {
      return Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      body: Stack(
        children: [
          CameraPreview(controller!),
          CustomPaint(
            painter: BoxPainter(boxes),
            child: Container(),
          ),
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
                  items: modelOptions
                      .map((model) => DropdownMenuItem(
                            value: model,
                            child: Text(model),
                          ))
                      .toList(),
                  onChanged: (val) {
                    setState(() => selectedModel = val!);
                  },
                ),
              ),
            ),
          )
        ],
      ),
    );
  }
}

class BoxPainter extends CustomPainter {
  final List<Rect> boxes;
  BoxPainter(this.boxes);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    for (final box in boxes) {
      canvas.drawRect(box, paint);
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}
