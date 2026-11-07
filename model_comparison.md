# Model Comparison: Deepfake Detection Architectures

| Model / Architecture | Dataset / Training Source | Input Size | Strengths | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **XceptionNet** | FaceForensics++ (c23/c40) | \ \times 299\$ | High accuracy on compressed manipulation artifacts; robust baseline. | Prone to false positives on heavy natural compression. |
| **EfficientNet (B4/B7)** | DFDC (DeepFake Detection Challenge) | \ \times 380\$ | Excellent parameter efficiency and generalization across diverse lighting. | Higher inference latency on edge devices. |
| **FaceForensics++ (FF++)** | Multi-source manipulated video dataset | \ \times 256\$ | Standard benchmark for blending boundary artifacts. | Limited performance on out-of-dataset face swaps. |
| **DeepFaceLab** | *Not a detector* (Synthesis tool) | Varies | Used exclusively for research and generation testing. | Inapplicable for direct detection tasks. |
