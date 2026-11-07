# Construction Site Safety PPE Detection

![PPE Detection Output](demo.gif)

This project aims to enhance safety on construction sites by detecting Personal Protective Equipment (PPE) using computer vision techniques. It utilizes YOLOv8, an object detection algorithm, to identify various PPE items such as hardhats, safety vests, and masks in real-time.

## Overview

Construction sites pose significant safety risks, and ensuring workers wear appropriate PPE is crucial for accident prevention. Traditional manual inspection methods can be time-consuming and error-prone. This project automates the process by leveraging computer vision to detect PPE compliance in real-time, enabling swift action to enforce safety protocols.

## Requirements

- Python 3.x
- ultralytics library
- OpenCV
- cvzone
- Custom dataset for training (obtained from Roboflow)

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/construction-site-safety.git
    cd construction-site-safety
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download pre-trained YOLOv8 weights (if not already downloaded) and place them in the specified directory.

## Usage

1. **Training:**

    Before using the model for detection, it needs to be trained on a custom dataset. Follow these steps:

    - Prepare your dataset and annotation files (e.g., in YOLO format).
    - Edit `train.py` to specify the path to your dataset and configuration.
    - Run the training script:

        ```bash
        python train.py
        ```

2. **Object Detection:**

    Once the model is trained, you can use it to detect PPE compliance in images or videos:

    - Replace the video file path in `main.py` with your input video.
    - Run the detection script:

        ```bash
        python main.py
        ```

    Detected PPE items will be highlighted with bounding boxes and labeled with their respective classes.

## Output

The output video (`output_video.mp4`) demonstrates the PPE detection process in action. Each frame is processed in real-time, and detected PPE items are annotated for easy identification.

## Results

The trained model achieves high accuracy in detecting various PPE items, including hardhats, safety vests, and masks. By automating the inspection process, it significantly reduces the time and effort required for ensuring safety compliance on construction sites.

## Future Improvements

- Fine-tuning the model with additional data to improve accuracy, especially in challenging scenarios.
- Integration with IoT devices for real-time monitoring and alerts.
- Deployment of the system on edge devices for on-site use.

## Contributors

- Joy Karmoker (https://github.com/JoyKarmoker)

## Acknowledgments

- Special thanks to Roboflow for providing the dataset and annotation tools.
- The ultralytics library and OpenCV community for their invaluable contributions to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.