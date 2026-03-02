<h2 align="center">
  DASE-DETR: Dynamic Adaptive Sparse Encoding for Dense Small Object Detection
</h2>

<p align="center">
    DASE-DETR (Dynamic Adaptive Sparse Encoding Detection Transformer) is an advanced framework designed to tackle dense small object detection in complex environments such as aerial imagery and UAV-based applications. This method efficiently reallocates computational resources to match semantic density, thereby improving detection accuracy while reducing computational redundancy. By using dynamic multi-scale saliency perception, adaptive sparse-dense transformations, and cross-scale semantic aggregation, DASE-DETR overcomes the limitations of traditional detection models that treat computational resources uniformly.
</p>

<p align="center">
  <b>Authors:</b> Dequan Zeng<sup>a</sup>, Peineng Yan<sup>a</sup>, Xuanyi Zhu<sup>b</sup>, Xinzhi Chen<sup>b</sup>, Mingxin Hou<sup>c,d</sup>, Jing Zhang<sup>c</sup>
</p>

<p align="center">
  <sup>a</sup>College of Shipbuilding and Shipping, Guangdong Ocean University, Zhanjiang, Guangdong Province, 524088, China<br>
  <sup>b</sup>School of Mathematics and Computer Science, Guangdong Ocean University, Zhanjiang, Guangdong Province, 524088, China<br>
  <sup>c</sup>School of Mechanical Engineering, Guangdong Ocean University, Zhanjiang, Guangdong Province, 524088, China<br>
  <sup>d</sup>Guangdong Provincial Key Laboratory of Intelligent Equipment for South China Sea Marine Ranching, Guangdong Ocean University, Zhanjiang, 524088, China
</p>

<p align="center">
  <b>∗ Corresponding Author:</b> Mingxin Hou<br>
  <b>Email:</b> <a href="mailto:houmx@gdou.edu.cn">houmx@gdou.edu.cn</a><br>
  <b>ORCID:</b> <a href="https://orcid.org/0000-0001-7751-4120">0000-0001-7751-4120</a>
</p>

---

## Experimental Results

### Comparison with Existing Methods on VisDrone2021 Validation Set

**Table 1:** Comparison of DASE-DETR with existing methods on the VisDrone2021 validation set. All models are trained under the same training protocol, with the number of training epochs set independently. Blue indicates improvement relative to the original baseline model.

#### Non-end-to-end Object Detectors

| Models | Epochs | Params(M) | GFLOPs | AP | AP50 | APS | APM | APL |
|--------|--------|-----------|--------|-----|------|-----|-----|-----|
| RTMDet-T [Arxiv 2022] | 300 | 4.9 | 8.0 | 18.4 | 31.2 | 7.7 | 28.8 | 36.7 |
| YOLOv11-S [Arxiv 2024] | 300 | 9.4 | 21.3 | 17.6 | 31.3 | 8.0 | 22.5 | 36.4 |
| YOLOv11-M [Arxiv 2024] | 300 | 20.1 | 68.5 | 20.3 | 35.0 | 9.8 | 31.2 | 41.3 |
| YOLOv12-S [Arxiv 2025] | 300 | 9.0 | 19.3 | 17.6 | 31.2 | 8.1 | 27.4 | 35.6 |
| YOLOv12-M [Arxiv 2025] | 300 | 19.7 | 60.4 | 19.2 | 33.6 | 9.4 | 29.8 | 38.6 |
| YOLOv13-S [Arxiv 2025] | 300 | 9.0 | 20.7 | 16.7 | 29.7 | 7.7 | 25.8 | 38.7 |
| FBRT-YOLO-S [AAAI 2025] | 300 | 2.9 | 22.9 | 18.3 | 32.3 | 8.5 | 28.3 | 42.5 |
| FBRT-YOLO-M [AAAI 2025] | 300 | 7.4 | 58.7 | 19.6 | 34.4 | 9.4 | 30.9 | 42.1 |

#### End-to-end Object Detectors

| Models | Epochs | Params(M) | GFLOPs | AP | AP50 | APS | APM | APL |
|--------|--------|-----------|--------|-----|------|-----|-----|-----|
| YOLOv10-S [Arxiv 2024] | 300 | 8.0 | 24.5 | 14.2 | 26.1 | 8.6 | 27.8 | 36.1 |
| YOLOv10-M [Arxiv 2024] | 300 | 16.5 | 63.5 | 17.9 | 32.3 | 9.7 | 30.0 | 41.4 |
| UAV-DETR-EV2 [Arxiv 2024] | 72 | 13 | 43 | 20.9 | 36.7 | 13.6 | 30.2 | 35.6 |
| RT-DETR-N [CVPR 2024] | 132 | 3.9 | 10.6 | 17.4 | 30.6 | 11.5 | 24.3 | 28.5 |
| DEIM-RT-DETR-N [CVPR 2025] | 132 | 3.9 | 10.6 | 19.4 | 33.8 | 13.2 | 27.1 | 30.8 |
| D-FINE-N [CVPR 2024] | 132 | 4.1 | 10.2 | 20.8 | 35.5 | 14.1 | 28.7 | 34.9 |
| DEIM-D-FINE-N [CVPR 2025] | 132 | 4.1 | 10.2 | 20.6 | 35.2 | 14.3 | 28.1 | 34.4 |
| **DASE-RT-DETR-N (Ours)** | **132** | **4.5** | **17.7** | **19.0** (+1.6) | **33.5** | **12.9** | **26.8** | **29.4** |
| **DASE-RT-DEIM-N (Ours)** | **132** | **4.5** | **17.7** | **20.9** (+1.5) | **36.1** | **14.2** | **28.8** | **32.8** |
| **DASE-DETR-N (Ours)** | **132** | **4.6** | **16.2** | **22.4** (+1.6) | **37.9** | **14.7** | **31.1** | **37.4** |
| **DASE-DEIM-N (Ours)** | **132** | **4.6** | **16.2** | **21.8** (+1.2) | **36.9** | **15.0** | **30.4** | **36.7** |

---
