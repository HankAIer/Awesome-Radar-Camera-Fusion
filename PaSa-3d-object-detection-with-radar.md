[PaSa search result](https://pasa-agent.ai/home?query=papers+about+3d+object+detection+with+radar&session=1741175840765413786)


```json
[
  {
    "link": "https://www.arxiv.org/abs/2003.00851v1",
    "title": "Deep Learning on Radar Centric 3D Object Detection",
    "publish_time": "20200227",
    "authors": [
      "Seungjun Lee"
    ],
    "abstract": "Even though many existing 3D object detection algorithms rely mostly on camera and LiDAR, camera and LiDAR are prone to be affected by harsh weather and lighting conditions. On the other hand, radar is resistant to such conditions. However, research has found only recently to apply deep neural networks on radar data. In this paper, we introduce a deep learning approach to 3D object detection with radar only. To the best of our knowledge, we are the first ones to demonstrate a deep learning-based 3D object detection model with radar only that was trained on the public radar dataset. To overcome the lack of radar labeled data, we propose a novel way of making use of abundant LiDAR data by transforming it into radar-like point cloud data and aggressive radar augmentation techniques.",
    "score": 0.9974769949913025
  },
  {
    "link": "https://www.arxiv.org/abs/2501.02314",
    "title": "RadarNeXt: Real-Time and Reliable 3D Object Detector Based On 4D mmWave Imaging Radar",
    "publish_time": "20250104",
    "authors": [
      "Liye Jia",
      "Runwei Guan",
      "Haocheng Zhao",
      "Qiuchi Zhao",
      "Ka Lok Man",
      "Jeremy Smith",
      "Limin Yu",
      "and Yutao Yue"
    ],
    "abstract": "3D object detection is crucial for Autonomous Driving (AD) and Advanced Driver Assistance Systems (ADAS). However, most 3D detectors prioritize detection accuracy, often overlooking network inference speed in practical applications. In this paper, we propose RadarNeXt, a real-time and reliable 3D object detector based on the 4D mmWave radar point clouds. It leverages the re-parameterizable neural networks to catch multi-scale features, reduce memory cost and accelerate the inference. Moreover, to highlight the irregular foreground features of radar point clouds and suppress background clutter, we propose a Multi-path Deformable Foreground Enhancement Network (MDFEN), ensuring detection accuracy while minimizing the sacrifice of speed and excessive number of parameters. Experimental results on View-of-Delft and TJ4DRadSet datasets validate the exceptional performance and efficiency of RadarNeXt, achieving 50.48 and 32.30 mAPs with the variant using our proposed MDFEN. Notably, our RadarNeXt variants achieve inference speeds of over 67.10 FPS on the RTX A4000 GPU and 28.40 FPS on the Jetson AGX Orin. This research demonstrates that RadarNeXt brings a novel and effective paradigm for 3D perception based on 4D mmWave radar.",
    "score": 0.997302770614624
  },
  {
    "link": "https://www.arxiv.org/abs/2403.15313",
    "title": "CR3DT: Camera-RADAR Fusion for 3D Detection and Tracking",
    "publish_time": "20240322",
    "authors": [
      "Nicolas Baumann",
      "Michael Baumgartner",
      "Edoardo Ghignone",
      "Jonas K\\\"uhne",
      "Tobias Fischer",
      "Yung-Hsu Yang",
      "Marc Pollefeys",
      "Michele Magno"
    ],
    "abstract": "Accurate detection and tracking of surrounding objects is essential to enable self-driving vehicles. While Light Detection and Ranging (LiDAR) sensors have set the benchmark for high performance, the appeal of camera-only solutions lies in their cost-effectiveness. Notably, despite the prevalent use of Radio Detection and Ranging (RADAR) sensors in automotive systems, their potential in 3D detection and tracking has been largely disregarded due to data sparsity and measurement noise. As a recent development, the combination of RADARs and cameras is emerging as a promising solution. This paper presents Camera-RADAR 3D Detection and Tracking (CR3DT), a camera-RADAR fusion model for 3D object detection, and Multi-Object Tracking (MOT). Building upon the foundations of the State-of-the-Art (SotA) camera-only BEVDet architecture, CR3DT demonstrates substantial improvements in both detection and tracking capabilities, by incorporating the spatial and velocity information of the RADAR sensor. Experimental results demonstrate an absolute improvement in detection performance of 5.3% in mean Average Precision (mAP) and a 14.9% increase in Average Multi-Object Tracking Accuracy (AMOTA) on the nuScenes dataset when leveraging both modalities. CR3DT bridges the gap between high-performance and cost-effective perception systems in autonomous driving, by capitalizing on the ubiquitous presence of RADAR in automotive applications.",
    "score": 0.9969724416732788
  },
  {
    "link": "https://www.arxiv.org/abs/2305.00397",
    "title": "TransCAR: Transformer-based Camera-And-Radar Fusion for 3D Object Detection",
    "publish_time": "20230430",
    "authors": [
      "Su Pang",
      "Daniel Morris",
      "Hayder Radha"
    ],
    "abstract": "Despite radar's popularity in the automotive industry, for fusion-based 3D object detection, most existing works focus on LiDAR and camera fusion. In this paper, we propose TransCAR, a Transformer-based Camera-And-Radar fusion solution for 3D object detection. Our TransCAR consists of two modules. The first module learns 2D features from surround-view camera images and then uses a sparse set of 3D object queries to index into these 2D features. The vision-updated queries then interact with each other via transformer self-attention layer. The second module learns radar features from multiple radar scans and then applies transformer decoder to learn the interactions between radar features and vision-updated queries. The cross-attention layer within the transformer decoder can adaptively learn the soft-association between the radar features and vision-updated queries instead of hard-association based on sensor calibration only. Finally, our model estimates a bounding box per query using set-to-set Hungarian loss, which enables the method to avoid non-maximum suppression. TransCAR improves the velocity estimation using the radar scans without temporal information. The superior experimental results of our TransCAR on the challenging nuScenes datasets illustrate that our TransCAR outperforms state-of-the-art Camera-Radar fusion-based 3D object detection approaches.",
    "score": 0.9965755343437195
  },
  {
    "link": "https://www.arxiv.org/abs/2105.00363",
    "title": "RADDet: Range-Azimuth-Doppler based Radar Object Detection for Dynamic Road Users",
    "publish_time": "20210502",
    "authors": [
      "Ao Zhang",
      "Farzan Erlik Nowruzi",
      "Robert Laganiere"
    ],
    "abstract": "Object detection using automotive radars has not been explored with deep learning models in comparison to the camera based approaches. This can be attributed to the lack of public radar datasets. In this paper, we collect a novel radar dataset that contains radar data in the form of Range-Azimuth-Doppler tensors along with the bounding boxes on the tensor for dynamic road users, category labels, and 2D bounding boxes on the Cartesian Bird-Eye-View range map. To build the dataset, we propose an instance-wise auto-annotation method. Furthermore, a novel Range-Azimuth-Doppler based multi-class object detection deep learning model is proposed. The algorithm is a one-stage anchor-based detector that generates both 3D bounding boxes and 2D bounding boxes on Range-Azimuth-Doppler and Cartesian domains, respectively. Our proposed algorithm achieves 56.3% AP with IOU of 0.3 on 3D bounding box predictions, and 51.6% with IOU of 0.5 on 2D bounding box prediction. Our dataset and the code can be found at https://github.com/ZhangAoCanada/RADDet.git.",
    "score": 0.9965597987174988
  },
  {
    "link": "https://www.arxiv.org/abs/2409.14751",
    "title": "UniBEVFusion: Unified Radar-Vision BEVFusion for 3D Object Detection",
    "publish_time": "20240923",
    "authors": [
      "Haocheng Zhao",
      "Runwei Guan",
      "Taoyu Wu",
      "Ka Lok Man",
      "Limin Yu",
      "Yutao Yue"
    ],
    "abstract": "4D millimeter-wave (MMW) radar, which provides both height information and dense point cloud data over 3D MMW radar, has become increasingly popular in 3D object detection. In recent years, radar-vision fusion models have demonstrated performance close to that of LiDAR-based models, offering advantages in terms of lower hardware costs and better resilience in extreme conditions. However, many radar-vision fusion models treat radar as a sparse LiDAR, underutilizing radar-specific information. Additionally, these multi-modal networks are often sensitive to the failure of a single modality, particularly vision. To address these challenges, we propose the Radar Depth Lift-Splat-Shoot (RDL) module, which integrates radar-specific data into the depth prediction process, enhancing the quality of visual Bird-Eye View (BEV) features. We further introduce a Unified Feature Fusion (UFF) approach that extracts BEV features across different modalities using shared module. To assess the robustness of multi-modal models, we develop a novel Failure Test (FT) ablation experiment, which simulates vision modality failure by injecting Gaussian noise. We conduct extensive experiments on the View-of-Delft (VoD) and TJ4D datasets. The results demonstrate that our proposed Unified BEVFusion (UniBEVFusion) network significantly outperforms state-of-the-art models on the TJ4D dataset, with improvements of 1.44 in 3D and 1.72 in BEV object detection accuracy.",
    "score": 0.9963489770889282
  },
  {
    "link": "https://www.arxiv.org/abs/2304.00670",
    "title": "CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception",
    "publish_time": "20230403",
    "authors": [
      "Youngseok Kim",
      "Juyeb Shin",
      "Sanmin Kim",
      "In-Jae Lee",
      "Jun Won Choi",
      "Dongsuk Kum"
    ],
    "abstract": "Autonomous driving requires an accurate and fast 3D perception system that includes 3D object detection, tracking, and segmentation. Although recent low-cost camera-based approaches have shown promising results, they are susceptible to poor illumination or bad weather conditions and have a large localization error. Hence, fusing camera with low-cost radar, which provides precise long-range measurement and operates reliably in all environments, is promising but has not yet been thoroughly investigated. In this paper, we propose Camera Radar Net (CRN), a novel camera-radar fusion framework that generates a semantically rich and spatially accurate bird's-eye-view (BEV) feature map for various tasks. To overcome the lack of spatial information in an image, we transform perspective view image features to BEV with the help of sparse but accurate radar points. We further aggregate image and radar feature maps in BEV using multi-modal deformable attention designed to tackle the spatial misalignment between inputs. CRN with real-time setting operates at 20 FPS while achieving comparable performance to LiDAR detectors on nuScenes, and even outperforms at a far distance on 100m setting. Moreover, CRN with offline setting yields 62.4% NDS, 57.5% mAP on nuScenes test set and ranks first among all camera and camera-radar 3D object detectors.",
    "score": 0.9963366985321045
  },
  {
    "link": "https://www.arxiv.org/abs/2210.09267",
    "title": "CramNet: Camera-Radar Fusion with Ray-Constrained Cross-Attention for Robust 3D Object Detection",
    "publish_time": "20221017",
    "authors": [
      "Jyh-Jing Hwang",
      "Henrik Kretzschmar",
      "Joshua Manela",
      "Sean Rafferty",
      "Nicholas Armstrong-Crews",
      "Tiffany Chen",
      "Dragomir Anguelov"
    ],
    "abstract": "Robust 3D object detection is critical for safe autonomous driving. Camera and radar sensors are synergistic as they capture complementary information and work well under different environmental conditions. Fusing camera and radar data is challenging, however, as each of the sensors lacks information along a perpendicular axis, that is, depth is unknown to camera and elevation is unknown to radar. We propose the camera-radar matching network CramNet, an efficient approach to fuse the sensor readings from camera and radar in a joint 3D space. To leverage radar range measurements for better camera depth predictions, we propose a novel ray-constrained cross-attention mechanism that resolves the ambiguity in the geometric correspondences between camera features and radar features. Our method supports training with sensor modality dropout, which leads to robust 3D object detection, even when a camera or radar sensor suddenly malfunctions on a vehicle. We demonstrate the effectiveness of our fusion approach through extensive experiments on the RADIATE dataset, one of the few large-scale datasets that provide radar radio frequency imagery. A camera-only variant of our method achieves competitive performance in monocular 3D object detection on the Waymo Open Dataset.",
    "score": 0.9960891008377075
  },
  {
    "link": "https://www.arxiv.org/abs/2307.10784",
    "title": "SMURF: Spatial Multi-Representation Fusion for 3D Object Detection with 4D Imaging Radar",
    "publish_time": "20230720",
    "authors": [
      "Jianan Liu",
      "Qiuchi Zhao",
      "Weiyi Xiong",
      "Tao Huang",
      "Qing-Long Han",
      "Bing Zhu"
    ],
    "abstract": "The 4D Millimeter wave (mmWave) radar is a promising technology for vehicle sensing due to its cost-effectiveness and operability in adverse weather conditions. However, the adoption of this technology has been hindered by sparsity and noise issues in radar point cloud data. This paper introduces spatial multi-representation fusion (SMURF), a novel approach to 3D object detection using a single 4D imaging radar. SMURF leverages multiple representations of radar detection points, including pillarization and density features of a multi-dimensional Gaussian mixture distribution through kernel density estimation (KDE). KDE effectively mitigates measurement inaccuracy caused by limited angular resolution and multi-path propagation of radar signals. Additionally, KDE helps alleviate point cloud sparsity by capturing density features. Experimental evaluations on View-of-Delft (VoD) and TJ4DRadSet datasets demonstrate the effectiveness and generalization ability of SMURF, outperforming recently proposed 4D imaging radar-based single-representation models. Moreover, while using 4D imaging radar only, SMURF still achieves comparable performance to the state-of-the-art 4D imaging radar and camera fusion-based method, with an increase of 1.22% in the mean average precision on bird's-eye view of TJ4DRadSet dataset and 1.32% in the 3D mean average precision on the entire annotated area of VoD dataset. Our proposed method demonstrates impressive inference time and addresses the challenges of real-time detection, with the inference time no more than 0.05 seconds for most scans on both datasets. This research highlights the benefits of 4D mmWave radar and is a strong benchmark for subsequent works regarding 3D object detection with 4D imaging radar.",
    "score": 0.9960839748382568
  },
  {
    "link": "https://www.arxiv.org/abs/2403.16440",
    "title": "RCBEVDet: Radar-camera Fusion in Bird's Eye View for 3D Object Detection",
    "publish_time": "20240325",
    "authors": [
      "Zhiwei Lin",
      "Zhe Liu",
      "Zhongyu Xia",
      "Xinhao Wang",
      "Yongtao Wang",
      "Shengxiang Qi",
      "Yang Dong",
      "Nan Dong",
      "Le Zhang",
      "Ce Zhu"
    ],
    "abstract": "Three-dimensional object detection is one of the key tasks in autonomous driving. To reduce costs in practice, low-cost multi-view cameras for 3D object detection are proposed to replace the expansive LiDAR sensors. However, relying solely on cameras is difficult to achieve highly accurate and robust 3D object detection. An effective solution to this issue is combining multi-view cameras with the economical millimeter-wave radar sensor to achieve more reliable multi-modal 3D object detection. In this paper, we introduce RCBEVDet, a radar-camera fusion 3D object detection method in the bird's eye view (BEV). Specifically, we first design RadarBEVNet for radar BEV feature extraction. RadarBEVNet consists of a dual-stream radar backbone and a Radar Cross-Section (RCS) aware BEV encoder. In the dual-stream radar backbone, a point-based encoder and a transformer-based encoder are proposed to extract radar features, with an injection and extraction module to facilitate communication between the two encoders. The RCS-aware BEV encoder takes RCS as the object size prior to scattering the point feature in BEV. Besides, we present the Cross-Attention Multi-layer Fusion module to automatically align the multi-modal BEV feature from radar and camera with the deformable attention mechanism, and then fuse the feature with channel and spatial fusion layers. Experimental results show that RCBEVDet achieves new state-of-the-art radar-camera fusion results on nuScenes and view-of-delft (VoD) 3D object detection benchmarks. Furthermore, RCBEVDet achieves better 3D detection results than all real-time camera-only and radar-camera 3D object detectors with a faster inference speed at 21~28 FPS. The source code will be released at https://github.com/VDIGPKU/RCBEVDet.",
    "score": 0.995868444442749
  },
  {
    "link": "https://www.arxiv.org/abs/2307.10249",
    "title": "RCM-Fusion: Radar-Camera Multi-Level Fusion for 3D Object Detection",
    "publish_time": "20230717",
    "authors": [
      "Jisong Kim",
      "Minjae Seong",
      "Geonho Bang",
      "Dongsuk Kum",
      "Jun Won Choi"
    ],
    "abstract": "While LiDAR sensors have been successfully applied to 3D object detection, the affordability of radar and camera sensors has led to a growing interest in fusing radars and cameras for 3D object detection. However, previous radar-camera fusion models were unable to fully utilize the potential of radar information. In this paper, we propose Radar-Camera Multi-level fusion (RCM-Fusion), which attempts to fuse both modalities at both feature and instance levels. For feature-level fusion, we propose a Radar Guided BEV Encoder which transforms camera features into precise BEV representations using the guidance of radar Bird's-Eye-View (BEV) features and combines the radar and camera BEV features. For instance-level fusion, we propose a Radar Grid Point Refinement module that reduces localization error by accounting for the characteristics of the radar point clouds. The experiments conducted on the public nuScenes dataset demonstrate that our proposed RCM-Fusion achieves state-of-the-art performances among single frame-based radar-camera fusion methods in the nuScenes 3D object detection benchmark. Code will be made publicly available.",
    "score": 0.995665967464447
  },
  {
    "link": "https://www.arxiv.org/abs/2411.19860v1",
    "title": "SpaRC: Sparse Radar-Camera Fusion for 3D Object Detection",
    "publish_time": "20241129",
    "authors": [
      "Philipp Wolters",
      "Johannes Gilg",
      "Torben Teepe",
      "Fabian Herzog",
      "Felix Fent",
      "Gerhard Rigoll"
    ],
    "abstract": "In this work, we present SpaRC, a novel Sparse fusion transformer for 3D perception that integrates multi-view image semantics with Radar and Camera point features. The fusion of radar and camera modalities has emerged as an efficient perception paradigm for autonomous driving systems. While conventional approaches utilize dense Bird's Eye View (BEV)-based architectures for depth estimation, contemporary query-based transformers excel in camera-only detection through object-centric methodology. However, these query-based approaches exhibit limitations in false positive detections and localization precision due to implicit depth modeling. We address these challenges through three key contributions: (1) sparse frustum fusion (SFF) for cross-modal feature alignment, (2) range-adaptive radar aggregation (RAR) for precise object localization, and (3) local self-attention (LSA) for focused query aggregation. In contrast to existing methods requiring computationally intensive BEV-grid rendering, SpaRC operates directly on encoded point features, yielding substantial improvements in efficiency and accuracy. Empirical evaluations on the nuScenes and TruckScenes benchmarks demonstrate that SpaRC significantly outperforms existing dense BEV-based and sparse query-based detectors. Our method achieves state-of-the-art performance metrics of 67.1 NDS and 63.1 AMOTA. The code and pretrained models are available at https://github.com/phi-wol/sparc.",
    "score": 0.9954603314399719
  },
  {
    "link": "https://www.arxiv.org/abs/2307.11323",
    "title": "HVDetFusion: A Simple and Robust Camera-Radar Fusion Framework",
    "publish_time": "20230721",
    "authors": [
      "Kai Lei",
      "Zhan Chen",
      "Shuman Jia",
      "Xiaoteng Zhang"
    ],
    "abstract": "In the field of autonomous driving, 3D object detection is a very important perception module. Although the current SOTA algorithm combines Camera and Lidar sensors, limited by the high price of Lidar, the current mainstream landing schemes are pure Camera sensors or Camera+Radar sensors. In this study, we propose a new detection algorithm called HVDetFusion, which is a multi-modal detection algorithm that not only supports pure camera data as input for detection, but also can perform fusion input of radar data and camera data. The camera stream does not depend on the input of Radar data, thus addressing the downside of previous methods. In the pure camera stream, we modify the framework of Bevdet4D for better perception and more efficient inference, and this stream has the whole 3D detection output. Further, to incorporate the benefits of Radar signals, we use the prior information of different object positions to filter the false positive information of the original radar data, according to the positioning information and radial velocity information recorded by the radar sensors to supplement and fuse the BEV features generated by the original camera data, and the effect is further improved in the process of fusion training. Finally, HVDetFusion achieves the new state-of-the-art 67.4\\% NDS on the challenging nuScenes test set among all camera-radar 3D object detectors. The code is available at https://github.com/HVXLab/HVDetFusion",
    "score": 0.9952915906906128
  },
  {
    "link": "https://www.arxiv.org/abs/2209.06535",
    "title": "CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer",
    "publish_time": "20220914",
    "authors": [
      "Youngseok Kim",
      "Sanmin Kim",
      "Jun Won Choi",
      "Dongsuk Kum"
    ],
    "abstract": "Camera and radar sensors have significant advantages in cost, reliability, and maintenance compared to LiDAR. Existing fusion methods often fuse the outputs of single modalities at the result-level, called the late fusion strategy. This can benefit from using off-the-shelf single sensor detection algorithms, but late fusion cannot fully exploit the complementary properties of sensors, thus having limited performance despite the huge potential of camera-radar fusion. Here we propose a novel proposal-level early fusion approach that effectively exploits both spatial and contextual properties of camera and radar for 3D object detection. Our fusion framework first associates image proposal with radar points in the polar coordinate system to efficiently handle the discrepancy between the coordinate system and spatial properties. Using this as a first stage, following consecutive cross-attention based feature fusion layers adaptively exchange spatio-contextual information between camera and radar, leading to a robust and attentive fusion. Our camera-radar fusion approach achieves the state-of-the-art 41.1% mAP and 52.3% NDS on the nuScenes test set, which is 8.7 and 10.8 points higher than the camera-only baseline, as well as yielding competitive performance on the LiDAR method.",
    "score": 0.9948691129684448
  },
  {
    "link": "https://www.arxiv.org/abs/2308.05478",
    "title": "Reviewing 3D Object Detectors in the Context of High-Resolution 3+1D Radar",
    "publish_time": "20230810",
    "authors": [
      "Patrick Palmer",
      "Martin Krueger",
      "Richard Altendorfer",
      "Ganesh Adam",
      "Torsten Bertram"
    ],
    "abstract": "Recent developments and the beginning market introduction of high-resolution imaging 4D (3+1D) radar sensors have initialized deep learning-based radar perception research. We investigate deep learning-based models operating on radar point clouds for 3D object detection. 3D object detection on lidar point cloud data is a mature area of 3D vision. Many different architectures have been proposed, each with strengths and weaknesses. Due to similarities between 3D lidar point clouds and 3+1D radar point clouds, those existing 3D object detectors are a natural basis to start deep learning-based 3D object detection on radar data. Thus, the first step is to analyze the detection performance of the existing models on the new data modality and evaluate them in depth. In order to apply existing 3D point cloud object detectors developed for lidar point clouds to the radar domain, they need to be adapted first. While some detectors, such as PointPillars, have already been adapted to be applicable to radar data, we have adapted others, e.g., Voxel R-CNN, SECOND, PointRCNN, and PV-RCNN. To this end, we conduct a cross-model validation (evaluating a set of models on one particular data set) as well as a cross-data set validation (evaluating all models in the model set on several data sets). The high-resolution radar data used are the View-of-Delft and Astyx data sets. Finally, we evaluate several adaptations of the models and their training procedures. We also discuss major factors influencing the detection performance on radar data and propose possible solutions indicating potential future research avenues.",
    "score": 0.9946730732917786
  },
  {
    "link": "https://www.arxiv.org/abs/2412.12725",
    "title": "RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion",
    "publish_time": "20241217",
    "authors": [
      "Xiaomeng Chu",
      "Jiajun Deng",
      "Guoliang You",
      "Yifan Duan",
      "Houqiang Li",
      "Yanyong Zhang"
    ],
    "abstract": "We propose Radar-Camera fusion transformer (RaCFormer) to boost the accuracy of 3D object detection by the following insight. The Radar-Camera fusion in outdoor 3D scene perception is capped by the image-to-BEV transformation--if the depth of pixels is not accurately estimated, the naive combination of BEV features actually integrates unaligned visual content. To avoid this problem, we propose a query-based framework that enables adaptively sample instance-relevant features from both the BEV and the original image view. Furthermore, we enhance system performance by two key designs: optimizing query initialization and strengthening the representational capacity of BEV. For the former, we introduce an adaptive circular distribution in polar coordinates to refine the initialization of object queries, allowing for a distance-based adjustment of query density. For the latter, we initially incorporate a radar-guided depth head to refine the transformation from image view to BEV. Subsequently, we focus on leveraging the Doppler effect of radar and introduce an implicit dynamic catcher to capture the temporal elements within the BEV. Extensive experiments on nuScenes and View-of-Delft (VoD) datasets validate the merits of our design. Remarkably, our method achieves superior results of 64.9% mAP and 70.2% NDS on nuScenes, even outperforming several LiDAR-based detectors. RaCFormer also secures the 1st ranking on the VoD dataset. The code will be released.",
    "score": 0.9944549202919006
  },
  {
    "link": "https://www.arxiv.org/abs/2011.04841",
    "title": "CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection",
    "publish_time": "20201110",
    "authors": [
      "Ramin Nabati",
      "Hairong Qi"
    ],
    "abstract": "The perception system in autonomous vehicles is responsible for detecting and tracking the surrounding objects. This is usually done by taking advantage of several sensing modalities to increase robustness and accuracy, which makes sensor fusion a crucial part of the perception system. In this paper, we focus on the problem of radar and camera sensor fusion and propose a middle-fusion approach to exploit both radar and camera data for 3D object detection. Our approach, called CenterFusion, first uses a center point detection network to detect objects by identifying their center points on the image. It then solves the key data association problem using a novel frustum-based method to associate the radar detections to their corresponding object's center point. The associated radar detections are used to generate radar-based feature maps to complement the image features, and regress to object properties such as depth, rotation and velocity. We evaluate CenterFusion on the challenging nuScenes dataset, where it improves the overall nuScenes Detection Score (NDS) of the state-of-the-art camera-based algorithm by more than 12%. We further show that CenterFusion significantly improves the velocity estimation accuracy without using any additional temporal information. The code is available at https://github.com/mrnabati/CenterFusion .",
    "score": 0.9944044351577759
  },
  {
    "link": "https://www.arxiv.org/abs/2003.01816",
    "title": "RODNet: Radar Object Detection Using Cross-Modal Supervision",
    "publish_time": "20200303",
    "authors": [
      "Yizhou Wang",
      "Zhongyu Jiang",
      "Xiangyu Gao",
      "Jenq-Neng Hwang",
      "Guanbin Xing",
      "Hui Liu"
    ],
    "abstract": "Radar is usually more robust than the camera in severe driving scenarios, e.g., weak/strong lighting and bad weather. However, unlike RGB images captured by a camera, the semantic information from the radar signals is noticeably difficult to extract. In this paper, we propose a deep radar object detection network (RODNet), to effectively detect objects purely from the carefully processed radar frequency data in the format of range-azimuth frequency heatmaps (RAMaps). Three different 3D autoencoder based architectures are introduced to predict object confidence distribution from each snippet of the input RAMaps. The final detection results are then calculated using our post-processing method, called location-based non-maximum suppression (L-NMS). Instead of using burdensome human-labeled ground truth, we train the RODNet using the annotations generated automatically by a novel 3D localization method using a camera-radar fusion (CRF) strategy. To train and evaluate our method, we build a new dataset -- CRUW, containing synchronized videos and RAMaps in various driving scenarios. After intensive experiments, our RODNet shows favorable object detection performance without the presence of the camera.",
    "score": 0.9943969249725342
  },
  {
    "link": "https://www.arxiv.org/abs/2208.12079",
    "title": "Bridging the View Disparity Between Radar and Camera Features for Multi-modal Fusion 3D Object Detection",
    "publish_time": "20220825",
    "authors": [
      "Taohua Zhou",
      "Yining Shi",
      "Junjie Chen",
      "Kun Jiang",
      "Mengmeng Yang",
      "Diange Yang"
    ],
    "abstract": "Environmental perception with the multi-modal fusion of radar and camera is crucial in autonomous driving to increase accuracy, completeness, and robustness. This paper focuses on utilizing millimeter-wave (MMW) radar and camera sensor fusion for 3D object detection. A novel method that realizes the feature-level fusion under the bird's-eye view (BEV) for a better feature representation is proposed. Firstly, radar points are augmented with temporal accumulation and sent to a spatial-temporal encoder for radar feature extraction. Meanwhile, multi-scale image 2D features which adapt to various spatial scales are obtained by image backbone and neck model. Then, image features are transformed to BEV with the designed view transformer. In addition, this work fuses the multi-modal features with a two-stage fusion model called point-fusion and ROI-fusion, respectively. Finally, a detection head regresses objects category and 3D locations. Experimental results demonstrate that the proposed method realizes the state-of-the-art (SOTA) performance under the most crucial detection metrics-mean average precision (mAP) and nuScenes detection score (NDS) on the challenging nuScenes dataset.",
    "score": 0.9936612844467163
  },
  {
    "link": "https://www.arxiv.org/abs/2302.10511",
    "title": "MVFusion: Multi-View 3D Object Detection with Semantic-aligned Radar and Camera Fusion",
    "publish_time": "20230221",
    "authors": [
      "Zizhang Wu",
      "Guilian Chen",
      "Yuanzhu Gan",
      "Lei Wang",
      "Jian Pu"
    ],
    "abstract": "Multi-view radar-camera fused 3D object detection provides a farther detection range and more helpful features for autonomous driving, especially under adverse weather. The current radar-camera fusion methods deliver kinds of designs to fuse radar information with camera data. However, these fusion approaches usually adopt the straightforward concatenation operation between multi-modal features, which ignores the semantic alignment with radar features and sufficient correlations across modals. In this paper, we present MVFusion, a novel Multi-View radar-camera Fusion method to achieve semantic-aligned radar features and enhance the cross-modal information interaction. To achieve so, we inject the semantic alignment into the radar features via the semantic-aligned radar encoder (SARE) to produce image-guided radar features. Then, we propose the radar-guided fusion transformer (RGFT) to fuse our radar and image features to strengthen the two modals' correlation from the global scope via the cross-attention mechanism. Extensive experiments show that MVFusion achieves state-of-the-art performance (51.7% NDS and 45.3% mAP) on the nuScenes dataset. We shall release our code and trained networks upon publication.",
    "score": 0.9925671815872192
  },
  {
    "link": "https://www.arxiv.org/abs/2408.05020",
    "title": "RadarPillars: Efficient Object Detection from 4D Radar Point Clouds",
    "publish_time": "20240809",
    "authors": [
      "Alexander Musiat",
      "Laurenz Reichardt",
      "Michael Schulze",
      "Oliver Wasenm\\\"uller"
    ],
    "abstract": "Automotive radar systems have evolved to provide not only range, azimuth and Doppler velocity, but also elevation data. This additional dimension allows for the representation of 4D radar as a 3D point cloud. As a result, existing deep learning methods for 3D object detection, which were initially developed for LiDAR data, are often applied to these radar point clouds. However, this neglects the special characteristics of 4D radar data, such as the extreme sparsity and the optimal utilization of velocity information. To address these gaps in the state-of-the-art, we present RadarPillars, a pillar-based object detection network. By decomposing radial velocity data, introducing PillarAttention for efficient feature extraction, and studying layer scaling to accommodate radar sparsity, RadarPillars significantly outperform state-of-the-art detection results on the View-of-Delft dataset. Importantly, this comes at a significantly reduced parameter count, surpassing existing methods in terms of efficiency and enabling real-time performance on edge devices.",
    "score": 0.9923682808876038
  },
  {
    "link": "https://www.arxiv.org/abs/2102.05150",
    "title": "RODNet: A Real-Time Radar Object Detection Network Cross-Supervised by Camera-Radar Fused Object 3D Localization",
    "publish_time": "20210209",
    "authors": [
      "Yizhou Wang",
      "Zhongyu Jiang",
      "Yudong Li",
      "Jenq-Neng Hwang",
      "Guanbin Xing",
      "Hui Liu"
    ],
    "abstract": "Various autonomous or assisted driving strategies have been facilitated through the accurate and reliable perception of the environment around a vehicle. Among the commonly used sensors, radar has usually been considered as a robust and cost-effective solution even in adverse driving scenarios, e.g., weak/strong lighting or bad weather. Instead of considering to fuse the unreliable information from all available sensors, perception from pure radar data becomes a valuable alternative that is worth exploring. In this paper, we propose a deep radar object detection network, named RODNet, which is cross-supervised by a camera-radar fused algorithm without laborious annotation efforts, to effectively detect objects from the radio frequency (RF) images in real-time. First, the raw signals captured by millimeter-wave radars are transformed to RF images in range-azimuth coordinates. Second, our proposed RODNet takes a sequence of RF images as the input to predict the likelihood of objects in the radar field of view (FoV). Two customized modules are also added to handle multi-chirp information and object relative motion. Instead of using human-labeled ground truth for training, the proposed RODNet is cross-supervised by a novel 3D localization of detected objects using a camera-radar fusion (CRF) strategy in the training stage. Finally, we propose a method to evaluate the object detection performance of the RODNet. Due to no existing public dataset available for our task, we create a new dataset, named CRUW, which contains synchronized RGB and RF image sequences in various driving scenarios. With intensive experiments, our proposed cross-supervised RODNet achieves 86% average precision and 88% average recall of object detection performance, which shows the robustness to noisy scenarios in various driving conditions.",
    "score": 0.9917787909507751
  },
  {
    "link": "https://www.arxiv.org/abs/2309.03734",
    "title": "ClusterFusion: Leveraging Radar Spatial Features for Radar-Camera 3D Object Detection in Autonomous Vehicles",
    "publish_time": "20230907",
    "authors": [
      "Irfan Tito Kurniawan",
      "Bambang Riyanto Trilaksono"
    ],
    "abstract": "Thanks to the complementary nature of millimeter wave radar and camera, deep learning-based radar-camera 3D object detection methods may reliably produce accurate detections even in low-visibility conditions. This makes them preferable to use in autonomous vehicles' perception systems, especially as the combined cost of both sensors is cheaper than the cost of a lidar. Recent radar-camera methods commonly perform feature-level fusion which often involves projecting the radar points onto the same plane as the image features and fusing the extracted features from both modalities. While performing fusion on the image plane is generally simpler and faster, projecting radar points onto the image plane flattens the depth dimension of the point cloud which might lead to information loss and makes extracting the spatial features of the point cloud harder. We proposed ClusterFusion, an architecture that leverages the local spatial features of the radar point cloud by clustering the point cloud and performing feature extraction directly on the point cloud clusters before projecting the features onto the image plane. ClusterFusion achieved the state-of-the-art performance among all radar-monocular camera methods on the test slice of the nuScenes dataset with 48.7% nuScenes detection score (NDS). We also investigated the performance of different radar feature extraction strategies on point cloud clusters: a handcrafted strategy, a learning-based strategy, and a combination of both, and found that the handcrafted strategy yielded the best performance. The main goal of this work is to explore the use of radar's local spatial and point-wise features by extracting them directly from radar point cloud clusters for a radar-monocular camera 3D object detection method that performs cross-modal feature fusion on the image plane.",
    "score": 0.9915817379951477
  },
  {
    "link": "https://www.arxiv.org/abs/2206.08171",
    "title": "K-Radar: 4D Radar Object Detection for Autonomous Driving in Various Weather Conditions",
    "publish_time": "20220616",
    "authors": [
      "Dong-Hee Paek",
      "Seung-Hyun Kong",
      "Kevin Tirta Wijaya"
    ],
    "abstract": "Unlike RGB cameras that use visible light bands (384$\\sim$769 THz) and Lidars that use infrared bands (361$\\sim$331 THz), Radars use relatively longer wavelength radio bands (77$\\sim$81 GHz), resulting in robust measurements in adverse weathers. Unfortunately, existing Radar datasets only contain a relatively small number of samples compared to the existing camera and Lidar datasets. This may hinder the development of sophisticated data-driven deep learning techniques for Radar-based perception. Moreover, most of the existing Radar datasets only provide 3D Radar tensor (3DRT) data that contain power measurements along the Doppler, range, and azimuth dimensions. As there is no elevation information, it is challenging to estimate the 3D bounding box of an object from 3DRT. In this work, we introduce KAIST-Radar (K-Radar), a novel large-scale object detection dataset and benchmark that contains 35K frames of 4D Radar tensor (4DRT) data with power measurements along the Doppler, range, azimuth, and elevation dimensions, together with carefully annotated 3D bounding box labels of objects on the roads. K-Radar includes challenging driving conditions such as adverse weathers (fog, rain, and snow) on various road structures (urban, suburban roads, alleyways, and highways). In addition to the 4DRT, we provide auxiliary measurements from carefully calibrated high-resolution Lidars, surround stereo cameras, and RTK-GPS. We also provide 4DRT-based object detection baseline neural networks (baseline NNs) and show that the height information is crucial for 3D object detection. And by comparing the baseline NN with a similarly-structured Lidar-based neural network, we demonstrate that 4D Radar is a more robust sensor for adverse weather conditions. All codes are available at https://github.com/kaist-avelab/k-radar.",
    "score": 0.9908960461616516
  },
  {
    "link": "https://www.arxiv.org/abs/2403.05061",
    "title": "RadarDistill: Boosting Radar-based Object Detection Performance via Knowledge Distillation from LiDAR Features",
    "publish_time": "20240308",
    "authors": [
      "Geonho Bang",
      "Kwangjin Choi",
      "Jisong Kim",
      "Dongsuk Kum",
      "Jun Won Choi"
    ],
    "abstract": "The inherent noisy and sparse characteristics of radar data pose challenges in finding effective representations for 3D object detection. In this paper, we propose RadarDistill, a novel knowledge distillation (KD) method, which can improve the representation of radar data by leveraging LiDAR data. RadarDistill successfully transfers desirable characteristics of LiDAR features into radar features using three key components: Cross-Modality Alignment (CMA), Activation-based Feature Distillation (AFD), and Proposal-based Feature Distillation (PFD). CMA enhances the density of radar features by employing multiple layers of dilation operations, effectively addressing the challenge of inefficient knowledge transfer from LiDAR to radar. AFD selectively transfers knowledge based on regions of the LiDAR features, with a specific focus on areas where activation intensity exceeds a predefined threshold. PFD similarly guides the radar network to selectively mimic features from the LiDAR network within the object proposals. Our comparative analyses conducted on the nuScenes datasets demonstrate that RadarDistill achieves state-of-the-art (SOTA) performance for radar-only object detection task, recording 20.5% in mAP and 43.7% in NDS. Also, RadarDistill significantly improves the performance of the camera-radar fusion model.",
    "score": 0.9896989464759827
  },
  {
    "link": "https://www.arxiv.org/abs/2105.05207",
    "title": "Rethinking of Radar's Role: A Camera-Radar Dataset and Systematic Annotator via Coordinate Alignment",
    "publish_time": "20210511",
    "authors": [
      "Yizhou Wang",
      "Gaoang Wang",
      "Hung-Min Hsu",
      "Hui Liu",
      "Jenq-Neng Hwang"
    ],
    "abstract": "Radar has long been a common sensor on autonomous vehicles for obstacle ranging and speed estimation. However, as a robust sensor to all-weather conditions, radar's capability has not been well-exploited, compared with camera or LiDAR. Instead of just serving as a supplementary sensor, radar's rich information hidden in the radio frequencies can potentially provide useful clues to achieve more complicated tasks, like object classification and detection. In this paper, we propose a new dataset, named CRUW, with a systematic annotator and performance evaluation system to address the radar object detection (ROD) task, which aims to classify and localize the objects in 3D purely from radar's radio frequency (RF) images. To the best of our knowledge, CRUW is the first public large-scale dataset with a systematic annotation and evaluation system, which involves camera RGB images and radar RF images, collected in various driving scenarios.",
    "score": 0.9887644052505493
  },
  {
    "link": "https://www.arxiv.org/abs/2310.16389",
    "title": "MVFAN: Multi-View Feature Assisted Network for 4D Radar Object Detection",
    "publish_time": "20231025",
    "authors": [
      "Qiao Yan",
      "Yihan Wang"
    ],
    "abstract": "4D radar is recognized for its resilience and cost-effectiveness under adverse weather conditions, thus playing a pivotal role in autonomous driving. While cameras and LiDAR are typically the primary sensors used in perception modules for autonomous vehicles, radar serves as a valuable supplementary sensor. Unlike LiDAR and cameras, radar remains unimpaired by harsh weather conditions, thereby offering a dependable alternative in challenging environments. Developing radar-based 3D object detection not only augments the competency of autonomous vehicles but also provides economic benefits. In response, we propose the Multi-View Feature Assisted Network (\\textit{MVFAN}), an end-to-end, anchor-free, and single-stage framework for 4D-radar-based 3D object detection for autonomous vehicles. We tackle the issue of insufficient feature utilization by introducing a novel Position Map Generation module to enhance feature learning by reweighing foreground and background points, and their features, considering the irregular distribution of radar point clouds. Additionally, we propose a pioneering backbone, the Radar Feature Assisted backbone, explicitly crafted to fully exploit the valuable Doppler velocity and reflectivity data provided by the 4D radar sensor. Comprehensive experiments and ablation studies carried out on Astyx and VoD datasets attest to the efficacy of our framework. The incorporation of Doppler velocity and RCS reflectivity dramatically improves the detection performance for small moving objects such as pedestrians and cyclists. Consequently, our approach culminates in a highly optimized 4D-radar-based 3D object detection capability for autonomous driving systems, setting a new standard in the field.",
    "score": 0.9885727167129517
  },
  {
    "link": "https://www.arxiv.org/abs/2204.13483v3",
    "title": "TJ4DRadSet: A 4D Radar Dataset for Autonomous Driving",
    "publish_time": "20220428",
    "authors": [
      "Lianqing Zheng",
      "Zhixiong Ma",
      "Xichan Zhu",
      "Bin Tan",
      "Sen Li",
      "Kai Long",
      "Weiqi Sun",
      "Sihan Chen",
      "Lu Zhang",
      "Mengyue Wan",
      "Libo Huang",
      "Jie Bai"
    ],
    "abstract": "The next-generation high-resolution automotive radar (4D radar) can provide additional elevation measurement and denser point clouds, which has great potential for 3D sensing in autonomous driving. In this paper, we introduce a dataset named TJ4DRadSet with 4D radar points for autonomous driving research. The dataset was collected in various driving scenarios, with a total of 7757 synchronized frames in 44 consecutive sequences, which are well annotated with 3D bounding boxes and track ids. We provide a 4D radar-based 3D object detection baseline for our dataset to demonstrate the effectiveness of deep learning methods for 4D radar point clouds. The dataset can be accessed via the following link: https://github.com/TJRadarLab/TJ4DRadSet.",
    "score": 0.9860433340072632
  },
  {
    "link": "https://www.arxiv.org/abs/2004.12165",
    "title": "CNN based Road User Detection using the 3D Radar Cube",
    "publish_time": "20200425",
    "authors": [
      "Andras Palffy",
      "Jiaao Dong",
      "Julian F. P. Kooij and Dariu M. Gavrila"
    ],
    "abstract": "This letter presents a novel radar based, single-frame, multi-class detection method for moving road users (pedestrian, cyclist, car), which utilizes low-level radar cube data. The method provides class information both on the radar target- and object-level. Radar targets are classified individually after extending the target features with a cropped block of the 3D radar cube around their positions, thereby capturing the motion of moving parts in the local velocity distribution. A Convolutional Neural Network (CNN) is proposed for this classification step. Afterwards, object proposals are generated with a clustering step, which not only considers the radar targets' positions and velocities, but their calculated class scores as well. In experiments on a real-life dataset we demonstrate that our method outperforms the state-of-the-art methods both target- and object-wise by reaching an average of 0.70 (baseline: 0.68) target-wise and 0.56 (baseline: 0.48) object-wise F1 score. Furthermore, we examine the importance of the used features in an ablation study.",
    "score": 0.9858464598655701
  },
  {
    "link": "https://www.arxiv.org/abs/2307.00724",
    "title": "LXL: LiDAR Excluded Lean 3D Object Detection with 4D Imaging Radar and Camera Fusion",
    "publish_time": "20230703",
    "authors": [
      "Weiyi Xiong",
      "Jianan Liu",
      "Tao Huang",
      "Qing-Long Han",
      "Yuxuan Xia",
      "Bing Zhu"
    ],
    "abstract": "As an emerging technology and a relatively affordable device, the 4D imaging radar has already been confirmed effective in performing 3D object detection in autonomous driving. Nevertheless, the sparsity and noisiness of 4D radar point clouds hinder further performance improvement, and in-depth studies about its fusion with other modalities are lacking. On the other hand, as a new image view transformation strategy, \"sampling\" has been applied in a few image-based detectors and shown to outperform the widely applied \"depth-based splatting\" proposed in Lift-Splat-Shoot (LSS), even without image depth prediction. However, the potential of \"sampling\" is not fully unleashed. This paper investigates the \"sampling\" view transformation strategy on the camera and 4D imaging radar fusion-based 3D object detection. LiDAR Excluded Lean (LXL) model, predicted image depth distribution maps and radar 3D occupancy grids are generated from image perspective view (PV) features and radar bird's eye view (BEV) features, respectively. They are sent to the core of LXL, called \"radar occupancy-assisted depth-based sampling\", to aid image view transformation. We demonstrated that more accurate view transformation can be performed by introducing image depths and radar information to enhance the \"sampling\" strategy. Experiments on VoD and TJ4DRadSet datasets show that the proposed method outperforms the state-of-the-art 3D object detection methods by a significant margin without bells and whistles. Ablation studies demonstrate that our method performs the best among different enhancement settings.",
    "score": 0.9845943450927734
  },
  {
    "link": "https://www.arxiv.org/abs/2209.12729",
    "title": "DeepFusion: A Robust and Modular 3D Object Detector for Lidars, Cameras and Radars",
    "publish_time": "20220926",
    "authors": [
      "Florian Drews",
      "Di Feng",
      "Florian Faion",
      "Lars Rosenbaum",
      "Michael Ulrich and Claudius Gl\\\"aser"
    ],
    "abstract": "We propose DeepFusion, a modular multi-modal architecture to fuse lidars, cameras and radars in different combinations for 3D object detection. Specialized feature extractors take advantage of each modality and can be exchanged easily, making the approach simple and flexible. Extracted features are transformed into bird's-eye-view as a common representation for fusion. Spatial and semantic alignment is performed prior to fusing modalities in the feature space. Finally, a detection head exploits rich multi-modal features for improved 3D detection performance. Experimental results for lidar-camera, lidar-camera-radar and camera-radar fusion show the flexibility and effectiveness of our fusion approach. In the process, we study the largely unexplored task of faraway car detection up to 225 meters, showing the benefits of our lidar-camera fusion. Furthermore, we investigate the required density of lidar points for 3D object detection and illustrate implications at the example of robustness against adverse weather conditions. Moreover, ablation studies on our camera-radar fusion highlight the importance of accurate depth estimation.",
    "score": 0.9825844168663025
  },
  {
    "link": "https://www.arxiv.org/abs/2211.06108",
    "title": "RaLiBEV: Radar and LiDAR BEV Fusion Learning for Anchor Box Free Object Detection Systems",
    "publish_time": "20221111",
    "authors": [
      "Yanlong Yang",
      "Jianan Liu",
      "Tao Huang",
      "Qing-Long Han",
      "Gang Ma and Bing Zhu"
    ],
    "abstract": "In autonomous driving, LiDAR and radar are crucial for environmental perception. LiDAR offers precise 3D spatial sensing information but struggles in adverse weather like fog. Conversely, radar signals can penetrate rain or mist due to their specific wavelength but are prone to noise disturbances. Recent state-of-the-art works reveal that the fusion of radar and LiDAR can lead to robust detection in adverse weather. The existing works adopt convolutional neural network architecture to extract features from each sensor data, then align and aggregate the two branch features to predict object detection results. However, these methods have low accuracy of predicted bounding boxes due to a simple design of label assignment and fusion strategies. In this paper, we propose a bird's-eye view fusion learning-based anchor box-free object detection system, which fuses the feature derived from the radar range-azimuth heatmap and the LiDAR point cloud to estimate possible objects. Different label assignment strategies have been designed to facilitate the consistency between the classification of foreground or background anchor points and the corresponding bounding box regressions. Furthermore, the performance of the proposed object detector is further enhanced by employing a novel interactive transformer module. The superior performance of the methods proposed in this paper has been demonstrated using the recently published Oxford Radar RobotCar dataset. Our system's average precision significantly outperforms the state-of-the-art method by 13.1% and 19.0% at Intersection of Union (IoU) of 0.8 under 'Clear+Foggy' training conditions for 'Clear' and 'Foggy' testing, respectively.",
    "score": 0.982582688331604
  },
  {
    "link": "https://www.arxiv.org/abs/2405.11682",
    "title": "FADet: A Multi-sensor 3D Object Detection Network based on Local Featured Attention",
    "publish_time": "20240519",
    "authors": [
      "Ziang Guo",
      "Zakhar Yagudin",
      "Selamawit Asfaw",
      "Artem Lykov",
      "Dzmitry Tsetserukou"
    ],
    "abstract": "Camera, LiDAR and radar are common perception sensors for autonomous driving tasks. Robust prediction of 3D object detection is optimally based on the fusion of these sensors. To exploit their abilities wisely remains a challenge because each of these sensors has its own characteristics. In this paper, we propose FADet, a multi-sensor 3D detection network, which specifically studies the characteristics of different sensors based on our local featured attention modules. For camera images, we propose dual-attention-based sub-module. For LiDAR point clouds, triple-attention-based sub-module is utilized while mixed-attention-based sub-module is applied for features of radar points. With local featured attention sub-modules, our FADet has effective detection results in long-tail and complex scenes from camera, LiDAR and radar input. On NuScenes validation dataset, FADet achieves state-of-the-art performance on LiDAR-camera object detection tasks with 71.8% NDS and 69.0% mAP, at the same time, on radar-camera object detection tasks with 51.7% NDS and 40.3% mAP. Code will be released at https://github.com/ZionGo6/FADet.",
    "score": 0.9823710918426514
  },
  {
    "link": "https://www.arxiv.org/abs/2309.17336",
    "title": "Robust 3D Object Detection from LiDAR-Radar Point Clouds via Cross-Modal Feature Augmentation",
    "publish_time": "20230929",
    "authors": [
      "Jianning Deng",
      "Gabriel Chan",
      "Hantao Zhong",
      "and Chris Xiaoxuan Lu"
    ],
    "abstract": "This paper presents a novel framework for robust 3D object detection from point clouds via cross-modal hallucination. Our proposed approach is agnostic to either hallucination direction between LiDAR and 4D radar. We introduce multiple alignments on both spatial and feature levels to achieve simultaneous backbone refinement and hallucination generation. Specifically, spatial alignment is proposed to deal with the geometry discrepancy for better instance matching between LiDAR and radar. The feature alignment step further bridges the intrinsic attribute gap between the sensing modalities and stabilizes the training. The trained object detection models can deal with difficult detection cases better, even though only single-modal data is used as the input during the inference stage. Extensive experiments on the View-of-Delft (VoD) dataset show that our proposed method outperforms the state-of-the-art (SOTA) methods for both radar and LiDAR object detection while maintaining competitive efficiency in runtime. Code is available at https://github.com/DJNing/See_beyond_seeing.",
    "score": 0.9806827306747437
  },
  {
    "link": "https://www.arxiv.org/abs/2203.10642",
    "title": "FUTR3D: A Unified Sensor Fusion Framework for 3D Detection",
    "publish_time": "20220320",
    "authors": [
      "Xuanyao Chen",
      "Tianyuan Zhang",
      "Yue Wang",
      "Yilun Wang",
      "Hang Zhao"
    ],
    "abstract": "Sensor fusion is an essential topic in many perception systems, such as autonomous driving and robotics. Existing multi-modal 3D detection models usually involve customized designs depending on the sensor combinations or setups. In this work, we propose the first unified end-to-end sensor fusion framework for 3D detection, named FUTR3D, which can be used in (almost) any sensor configuration. FUTR3D employs a query-based Modality-Agnostic Feature Sampler (MAFS), together with a transformer decoder with a set-to-set loss for 3D detection, thus avoiding using late fusion heuristics and post-processing tricks. We validate the effectiveness of our framework on various combinations of cameras, low-resolution LiDARs, high-resolution LiDARs, and Radars. On NuScenes dataset, FUTR3D achieves better performance over specifically designed methods across different sensor combinations. Moreover, FUTR3D achieves great flexibility with different sensor configurations and enables low-cost autonomous driving. For example, only using a 4-beam LiDAR with cameras, FUTR3D (58.0 mAP) achieves on par performance with state-of-the-art 3D detection model CenterPoint (56.6 mAP) using a 32-beam LiDAR.",
    "score": 0.9652471542358398
  },
  {
    "link": "https://www.arxiv.org/abs/2203.04440",
    "title": "Pointillism: Accurate 3D bounding box estimation with multi-radars",
    "publish_time": "20220308",
    "authors": [
      "Kshitiz Bansal",
      "Keshav Rungta",
      "Siyuan Zhu",
      "Dinesh Bharadia"
    ],
    "abstract": "Autonomous perception requires high-quality environment sensing in the form of 3D bounding boxes of dynamic objects. The primary sensors used in automotive systems are light-based cameras and LiDARs. However, they are known to fail in adverse weather conditions. Radars can potentially solve this problem as they are barely affected by adverse weather conditions. However, specular reflections of wireless signals cause poor performance of radar point clouds. We introduce Pointillism, a system that combines data from multiple spatially separated radars with an optimal separation to mitigate these problems. We introduce a novel concept of Cross Potential Point Clouds, which uses the spatial diversity induced by multiple radars and solves the problem of noise and sparsity in radar point clouds. Furthermore, we present the design of RP-net, a novel deep learning architecture, designed explicitly for radar's sparse data distribution, to enable accurate 3D bounding box estimation. The spatial techniques designed and proposed in this paper are fundamental to radars point cloud distribution and would benefit other radar sensing applications.",
    "score": 0.9613655805587769
  },
  {
    "link": "https://www.arxiv.org/abs/2408.06772",
    "title": "Exploring Domain Shift on Radar-Based 3D Object Detection Amidst Diverse Environmental Conditions",
    "publish_time": "20240813",
    "authors": [
      "Miao Zhang",
      "Sherif Abdulatif",
      "Benedikt Loesch",
      "Marco Altmann",
      "Marius Schwarz and Bin Yang"
    ],
    "abstract": "The rapid evolution of deep learning and its integration with autonomous driving systems have led to substantial advancements in 3D perception using multimodal sensors. Notably, radar sensors show greater robustness compared to cameras and lidar under adverse weather and varying illumination conditions. This study delves into the often-overlooked yet crucial issue of domain shift in 4D radar-based object detection, examining how varying environmental conditions, such as different weather patterns and road types, impact 3D object detection performance. Our findings highlight distinct domain shifts across various weather scenarios, revealing unique dataset sensitivities that underscore the critical role of radar point cloud generation. Additionally, we demonstrate that transitioning between different road types, especially from highways to urban settings, introduces notable domain shifts, emphasizing the necessity for diverse data collection across varied road environments. To the best of our knowledge, this is the first comprehensive analysis of domain shift effects on 4D radar-based object detection. We believe this empirical study contributes to understanding the complex nature of domain shifts in radar data and suggests paths forward for data collection strategy in the face of environmental variability.",
    "score": 0.9512742161750793
  },
  {
    "link": "https://www.arxiv.org/abs/2403.07746",
    "title": "Unleashing HyDRa: Hybrid Fusion, Depth Consistency and Radar for Unified 3D Perception",
    "publish_time": "20240312",
    "authors": [
      "Philipp Wolters",
      "Johannes Gilg",
      "Torben Teepe",
      "Fabian Herzog",
      "Anouar Laouichi",
      "Martin Hofmann",
      "Gerhard Rigoll"
    ],
    "abstract": "Low-cost, vision-centric 3D perception systems for autonomous driving have made significant progress in recent years, narrowing the gap to expensive LiDAR-based methods. The primary challenge in becoming a fully reliable alternative lies in robust depth prediction capabilities, as camera-based systems struggle with long detection ranges and adverse lighting and weather conditions. In this work, we introduce HyDRa, a novel camera-radar fusion architecture for diverse 3D perception tasks. Building upon the principles of dense BEV (Bird's Eye View)-based architectures, HyDRa introduces a hybrid fusion approach to combine the strengths of complementary camera and radar features in two distinct representation spaces. Our Height Association Transformer module leverages radar features already in the perspective view to produce more robust and accurate depth predictions. In the BEV, we refine the initial sparse representation by a Radar-weighted Depth Consistency. HyDRa achieves a new state-of-the-art for camera-radar fusion of 64.2 NDS (+1.8) and 58.4 AMOTA (+1.5) on the public nuScenes dataset. Moreover, our new semantically rich and spatially accurate BEV features can be directly converted into a powerful occupancy representation, beating all previous camera-based methods on the Occ3D benchmark by an impressive 3.7 mIoU. Code and models are available at https://github.com/phi-wol/hydra.",
    "score": 0.9453081488609314
  },
  {
    "link": "https://www.arxiv.org/abs/2205.02111v2",
    "title": "Improved Orientation Estimation and Detection with Hybrid Object Detection Networks for Automotive Radar",
    "publish_time": "20220503",
    "authors": [
      "Michael Ulrich",
      "Sascha Braun",
      "Daniel Köhler",
      "Daniel Niederlöhner",
      "Florian Faion",
      "Claudius Gläser",
      "Holger Blume"
    ],
    "abstract": "This paper presents novel hybrid architectures that combine grid- and point-based processing to improve the detection performance and orientation estimation of radar-based object detection networks. Purely grid-based detection models operate on a bird's-eye-view (BEV) projection of the input point cloud. These approaches suffer from a loss of detailed information through the discrete grid resolution. This applies in particular to radar object detection, where relatively coarse grid resolutions are commonly used to account for the sparsity of radar point clouds. In contrast, point-based models are not affected by this problem as they process point clouds without discretization. However, they generally exhibit worse detection performances than grid-based methods. We show that a point-based model can extract neighborhood features, leveraging the exact relative positions of points, before grid rendering. This has significant benefits for a subsequent grid-based convolutional detection backbone. In experiments on the public nuScenes dataset our hybrid architecture achieves improvements in terms of detection performance (19.7% higher mAP for car class than next-best radar-only submission) and orientation estimates (11.5% relative orientation improvement) over networks from previous literature.",
    "score": 0.9449592232704163
  },
  {
    "link": "https://www.arxiv.org/abs/2007.14366",
    "title": "RadarNet: Exploiting Radar for Robust Perception of Dynamic Objects",
    "publish_time": "20200728",
    "authors": [
      "Bin Yang",
      "Runsheng Guo",
      "Ming Liang",
      "Sergio Casas",
      "Raquel Urtasun"
    ],
    "abstract": "We tackle the problem of exploiting Radar for perception in the context of self-driving as Radar provides complementary information to other sensors such as LiDAR or cameras in the form of Doppler velocity. The main challenges of using Radar are the noise and measurement ambiguities which have been a struggle for existing simple input or output fusion methods. To better address this, we propose a new solution that exploits both LiDAR and Radar sensors for perception. Our approach, dubbed RadarNet, features a voxel-based early fusion and an attention-based late fusion, which learn from data to exploit both geometric and dynamic information of Radar data. RadarNet achieves state-of-the-art results on two large-scale real-world datasets in the tasks of object detection and velocity estimation. We further show that exploiting Radar improves the perception capabilities of detecting faraway objects and understanding the motion of dynamic objects.",
    "score": 0.938634991645813
  },
  {
    "link": "https://www.arxiv.org/abs/2307.16532",
    "title": "Echoes Beyond Points: Unleashing the Power of Raw Radar Data in Multi-modality Fusion",
    "publish_time": "20230731",
    "authors": [
      "Yang Liu",
      "Feng Wang",
      "Naiyan Wang",
      "Zhaoxiang Zhang"
    ],
    "abstract": "Radar is ubiquitous in autonomous driving systems due to its low cost and good adaptability to bad weather. Nevertheless, the radar detection performance is usually inferior because its point cloud is sparse and not accurate due to the poor azimuth and elevation resolution. Moreover, point cloud generation algorithms already drop weak signals to reduce the false targets which may be suboptimal for the use of deep fusion. In this paper, we propose a novel method named EchoFusion to skip the existing radar signal processing pipeline and then incorporate the radar raw data with other sensors. Specifically, we first generate the Bird's Eye View (BEV) queries and then take corresponding spectrum features from radar to fuse with other sensors. By this approach, our method could utilize both rich and lossless distance and speed clues from radar echoes and rich semantic clues from images, making our method surpass all existing methods on the RADIal dataset, and approach the performance of LiDAR. The code will be released on https://github.com/tusen-ai/EchoFusion.",
    "score": 0.8920864462852478
  },
  {
    "link": "https://www.arxiv.org/abs/2212.11172",
    "title": "A recurrent CNN for online object detection on raw radar frames",
    "publish_time": "20221221",
    "authors": [
      "Colin Decourt",
      "Rufin VanRullen",
      "Didier Salle and Thomas Oberlin"
    ],
    "abstract": "Automotive radar sensors provide valuable information for advanced driving assistance systems (ADAS). Radars can reliably estimate the distance to an object and the relative velocity, regardless of weather and light conditions. However, radar sensors suffer from low resolution and huge intra-class variations in the shape of objects. Exploiting the time information (e.g., multiple frames) has been shown to help to capture better the dynamics of objects and, therefore, the variation in the shape of objects. Most temporal radar object detectors use 3D convolutions to learn spatial and temporal information. However, these methods are often non-causal and unsuitable for real-time applications. This work presents RECORD, a new recurrent CNN architecture for online radar object detection. We propose an end-to-end trainable architecture mixing convolutions and ConvLSTMs to learn spatio-temporal dependencies between successive frames. Our model is causal and requires only the past information encoded in the memory of the ConvLSTMs to detect objects. Our experiments show such a method's relevance for detecting objects in different radar representations (range-Doppler, range-angle) and outperform state-of-the-art models on the ROD2021 and CARRADA datasets while being less computationally expensive.",
    "score": 0.8795737028121948
  },
  {
    "link": "https://www.arxiv.org/abs/2206.07959",
    "title": "Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?",
    "publish_time": "20220616",
    "authors": [
      "Adam W. Harley",
      "Zhaoyuan Fang",
      "Jie Li",
      "Rares Ambrus",
      "Katerina Fragkiadaki"
    ],
    "abstract": "Building 3D perception systems for autonomous vehicles that do not rely on high-density LiDAR is a critical research problem because of the expense of LiDAR systems compared to cameras and other sensors. Recent research has developed a variety of camera-only methods, where features are differentiably \"lifted\" from the multi-camera images onto the 2D ground plane, yielding a \"bird's eye view\" (BEV) feature representation of the 3D space around the vehicle. This line of work has produced a variety of novel \"lifting\" methods, but we observe that other details in the training setups have shifted at the same time, making it unclear what really matters in top-performing methods. We also observe that using cameras alone is not a real-world constraint, considering that additional sensors like radar have been integrated into real vehicles for years already. In this paper, we first of all attempt to elucidate the high-impact factors in the design and training protocol of BEV perception models. We find that batch size and input resolution greatly affect performance, while lifting strategies have a more modest effect -- even a simple parameter-free lifter works well. Second, we demonstrate that radar data can provide a substantial boost to performance, helping to close the gap between camera-only and LiDAR-enabled systems. We analyze the radar usage details that lead to good performance, and invite the community to re-consider this commonly-neglected part of the sensor platform.",
    "score": 0.8508093953132629
  },
  {
    "link": "https://www.arxiv.org/abs/2408.00565",
    "title": "MUFASA: Multi-View Fusion and Adaptation Network with Spatial Awareness for Radar Object Detection",
    "publish_time": "20240801",
    "authors": [
      "Xiangyuan Peng",
      "Miao Tang",
      "Huawei Sun",
      "Kay Bierzynski",
      "Lorenzo Servadei",
      "and Robert Wille"
    ],
    "abstract": "In recent years, approaches based on radar object detection have made significant progress in autonomous driving systems due to their robustness under adverse weather compared to LiDAR. However, the sparsity of radar point clouds poses challenges in achieving precise object detection, highlighting the importance of effective and comprehensive feature extraction technologies. To address this challenge, this paper introduces a comprehensive feature extraction method for radar point clouds. This study first enhances the capability of detection networks by using a plug-and-play module, GeoSPA. It leverages the Lalonde features to explore local geometric patterns. Additionally, a distributed multi-view attention mechanism, DEMVA, is designed to integrate the shared information across the entire dataset with the global information of each individual frame. By employing the two modules, we present our method, MUFASA, which enhances object detection performance through improved feature extraction. The approach is evaluated on the VoD and TJ4DRaDSet datasets to demonstrate its effectiveness. In particular, we achieve state-of-the-art results among radar-based methods on the VoD dataset with the mAP of 50.24%.",
    "score": 0.8506647348403931
  },
  {
    "link": "https://www.arxiv.org/abs/1903.11027",
    "title": "nuScenes: A multimodal dataset for autonomous driving",
    "publish_time": "20190326",
    "authors": [
      "Holger Caesar",
      "Varun Bankiti",
      "Alex H. Lang",
      "Sourabh Vora",
      "Venice Erin Liong",
      "Qiang Xu",
      "Anush Krishnan",
      "Yu Pan",
      "Giancarlo Baldan",
      "Oscar Beijbom"
    ],
    "abstract": "Robust detection and tracking of objects is crucial for the deployment of autonomous vehicle technology. Image based benchmark datasets have driven development in computer vision tasks such as object detection, tracking and segmentation of agents in the environment. Most autonomous vehicles, however, carry a combination of cameras and range sensors such as lidar and radar. As machine learning based methods for detection and tracking become more prevalent, there is a need to train and evaluate such methods on datasets containing range sensor data along with images. In this work we present nuTonomy scenes (nuScenes), the first dataset to carry the full autonomous vehicle sensor suite: 6 cameras, 5 radars and 1 lidar, all with full 360 degree field of view. nuScenes comprises 1000 scenes, each 20s long and fully annotated with 3D bounding boxes for 23 classes and 8 attributes. It has 7x as many annotations and 100x as many images as the pioneering KITTI dataset. We define novel 3D detection and tracking metrics. We also provide careful dataset analysis as well as baselines for lidar and image based detection and tracking. Data, development kit and more information are available online.",
    "score": 0.8345030546188354
  },
  {
    "link": "https://www.arxiv.org/abs/2009.08428",
    "title": "Radar-Camera Sensor Fusion for Joint Object Detection and Distance Estimation in Autonomous Vehicles",
    "publish_time": "20200917",
    "authors": [
      "Ramin Nabati",
      "Hairong Qi"
    ],
    "abstract": "In this paper we present a novel radar-camera sensor fusion framework for accurate object detection and distance estimation in autonomous driving scenarios. The proposed architecture uses a middle-fusion approach to fuse the radar point clouds and RGB images. Our radar object proposal network uses radar point clouds to generate 3D proposals from a set of 3D prior boxes. These proposals are mapped to the image and fed into a Radar Proposal Refinement (RPR) network for objectness score prediction and box refinement. The RPR network utilizes both radar information and image feature maps to generate accurate object proposals and distance estimations. The radar-based proposals are combined with image-based proposals generated by a modified Region Proposal Network (RPN). The RPN has a distance regression layer for estimating distance for every generated proposal. The radar-based and image-based proposals are merged and used in the next stage for object classification. Experiments on the challenging nuScenes dataset show our method outperforms other existing radar-camera fusion methods in the 2D object detection task while at the same time accurately estimates objects' distances.",
    "score": 0.8343639969825745
  },
  {
    "link": "https://www.arxiv.org/abs/2004.05310",
    "title": "Probabilistic Oriented Object Detection in Automotive Radar",
    "publish_time": "20200411",
    "authors": [
      "Xu Dong",
      "Pengluo Wang",
      "Pengyue Zhang",
      "Langechuan Liu"
    ],
    "abstract": "Autonomous radar has been an integral part of advanced driver assistance systems due to its robustness to adverse weather and various lighting conditions. Conventional automotive radars use digital signal processing (DSP) algorithms to process raw data into sparse radar pins that do not provide information regarding the size and orientation of the objects. In this paper, we propose a deep-learning based algorithm for radar object detection. The algorithm takes in radar data in its raw tensor representation and places probabilistic oriented bounding boxes around the detected objects in bird's-eye-view space. We created a new multimodal dataset with 102544 frames of raw radar and synchronized LiDAR data. To reduce human annotation effort we developed a scalable pipeline to automatically annotate ground truth using LiDAR as reference. Based on this dataset we developed a vehicle detection pipeline using raw radar data as the only input. Our best performing radar detection model achieves 77.28\\% AP under oriented IoU of 0.3. To the best of our knowledge, this is the first attempt to investigate object detection with raw radar data for conventional corner automotive radars.",
    "score": 0.8341743350028992
  },
  {
    "link": "https://www.arxiv.org/abs/2107.05150",
    "title": "CFTrack: Center-based Radar and Camera Fusion for 3D Multi-Object Tracking",
    "publish_time": "20210711",
    "authors": [
      "Ramin Nabati",
      "Landon Harris",
      "Hairong Qi"
    ],
    "abstract": "3D multi-object tracking is a crucial component in the perception system of autonomous driving vehicles. Tracking all dynamic objects around the vehicle is essential for tasks such as obstacle avoidance and path planning. Autonomous vehicles are usually equipped with different sensor modalities to improve accuracy and reliability. While sensor fusion has been widely used in object detection networks in recent years, most existing multi-object tracking algorithms either rely on a single input modality, or do not fully exploit the information provided by multiple sensing modalities. In this work, we propose an end-to-end network for joint object detection and tracking based on radar and camera sensor fusion. Our proposed method uses a center-based radar-camera fusion algorithm for object detection and utilizes a greedy algorithm for object association. The proposed greedy algorithm uses the depth, velocity and 2D displacement of the detected objects to associate them through time. This makes our tracking algorithm very robust to occluded and overlapping objects, as the depth and velocity information can help the network in distinguishing them. We evaluate our method on the challenging nuScenes dataset, where it achieves 20.0 AMOTA and outperforms all vision-based 3D tracking methods in the benchmark, as well as the baseline LiDAR-based method. Our method is online with a runtime of 35ms per image, making it very suitable for autonomous driving applications.",
    "score": 0.7762803435325623
  },
  {
    "link": "https://www.arxiv.org/abs/2303.02203",
    "title": "X$^3$KD: Knowledge Distillation Across Modalities, Tasks and Stages for Multi-Camera 3D Object Detection",
    "publish_time": "20230303",
    "authors": [
      "Marvin Klingner",
      "Shubhankar Borse",
      "Varun Ravi Kumar",
      "Behnaz Rezaei",
      "Venkatraman Narayanan",
      "Senthil Yogamani",
      "Fatih Porikli"
    ],
    "abstract": "Recent advances in 3D object detection (3DOD) have obtained remarkably strong results for LiDAR-based models. In contrast, surround-view 3DOD models based on multiple camera images underperform due to the necessary view transformation of features from perspective view (PV) to a 3D world representation which is ambiguous due to missing depth information. This paper introduces X$^3$KD, a comprehensive knowledge distillation framework across different modalities, tasks, and stages for multi-camera 3DOD. Specifically, we propose cross-task distillation from an instance segmentation teacher (X-IS) in the PV feature extraction stage providing supervision without ambiguous error backpropagation through the view transformation. After the transformation, we apply cross-modal feature distillation (X-FD) and adversarial training (X-AT) to improve the 3D world representation of multi-camera features through the information contained in a LiDAR-based 3DOD teacher. Finally, we also employ this teacher for cross-modal output distillation (X-OD), providing dense supervision at the prediction stage. We perform extensive ablations of knowledge distillation at different stages of multi-camera 3DOD. Our final X$^3$KD model outperforms previous state-of-the-art approaches on the nuScenes and Waymo datasets and generalizes to RADAR-based 3DOD. Qualitative results video at https://youtu.be/1do9DPFmr38.",
    "score": 0.7048438787460327
  },
  {
    "link": "https://www.arxiv.org/abs/2406.01011v1",
    "title": "Multi-Object Tracking based on Imaging Radar 3D Object Detection",
    "publish_time": "20240603",
    "authors": [
      "Patrick Palmer",
      "Martin Krüger",
      "Richard Altendorfer",
      "Torsten Bertram"
    ],
    "abstract": "Effective tracking of surrounding traffic participants allows for an accurate state estimation as a necessary ingredient for prediction of future behavior and therefore adequate planning of the ego vehicle trajectory. One approach for detecting and tracking surrounding traffic participants is the combination of a learning based object detector with a classical tracking algorithm. Learning based object detectors have been shown to work adequately on lidar and camera data, while learning based object detectors using standard radar data input have proven to be inferior. Recently, with the improvements to radar sensor technology in the form of imaging radars, the object detection performance on radar was greatly improved but is still limited compared to lidar sensors due to the sparsity of the radar point cloud. This presents a unique challenge for the task of multi-object tracking. The tracking algorithm must overcome the limited detection quality while generating consistent tracks. To this end, a comparison between different multi-object tracking methods on imaging radar data is required to investigate its potential for downstream tasks. The work at hand compares multiple approaches and analyzes their limitations when applied to imaging radar data. Furthermore, enhancements to the presented approaches in the form of probabilistic association algorithms are considered for this task.",
    "score": 0.6784406304359436
  },
  {
    "link": "https://www.arxiv.org/abs/2406.00714",
    "title": "A Survey of Deep Learning Based Radar and Vision Fusion for 3D Object Detection in Autonomous Driving",
    "publish_time": "20240602",
    "authors": [
      "Di Wu",
      "Feng Yang",
      "Benlian Xu",
      "Pan Liao and Bo Liu"
    ],
    "abstract": "With the rapid advancement of autonomous driving technology, there is a growing need for enhanced safety and efficiency in the automatic environmental perception of vehicles during their operation. In modern vehicle setups, cameras and mmWave radar (radar), being the most extensively employed sensors, demonstrate complementary characteristics, inherently rendering them conducive to fusion and facilitating the achievement of both robust performance and cost-effectiveness. This paper focuses on a comprehensive survey of radar-vision (RV) fusion based on deep learning methods for 3D object detection in autonomous driving. We offer a comprehensive overview of each RV fusion category, specifically those employing region of interest (ROI) fusion and end-to-end fusion strategies. As the most promising fusion strategy at present, we provide a deeper classification of end-to-end fusion methods, including those 3D bounding box prediction based and BEV based approaches. Moreover, aligning with recent advancements, we delineate the latest information on 4D radar and its cutting-edge applications in autonomous vehicles (AVs). Finally, we present the possible future trends of RV fusion and summarize this paper.",
    "score": 0.650687038898468
  },
  {
    "link": "https://www.arxiv.org/abs/2112.10646",
    "title": "Raw High-Definition Radar for Multi-Task Learning",
    "publish_time": "20211220",
    "authors": [
      "Julien Rebut",
      "Arthur Ouaknine",
      "Waqas Malik and Patrick P\\'erez"
    ],
    "abstract": "With their robustness to adverse weather conditions and ability to measure speeds, radar sensors have been part of the automotive landscape for more than two decades. Recent progress toward High Definition (HD) Imaging radar has driven the angular resolution below the degree, thus approaching laser scanning performance. However, the amount of data a HD radar delivers and the computational cost to estimate the angular positions remain a challenge. In this paper, we propose a novel HD radar sensing model, FFT-RadNet, that eliminates the overhead of computing the range-azimuth-Doppler 3D tensor, learning instead to recover angles from a range-Doppler spectrum. FFT-RadNet is trained both to detect vehicles and to segment free driving space. On both tasks, it competes with the most recent radar-based models while requiring less compute and memory. Also, we collected and annotated 2-hour worth of raw data from synchronized automotive-grade sensors (camera, laser, HD radar) in various environments (city street, highway, countryside road). This unique dataset, nick-named RADIal for \"Radar, Lidar et al.\", is available at https://github.com/valeoai/RADIal.",
    "score": 0.6505876779556274
  },
  {
    "link": "https://www.arxiv.org/abs/2011.08981",
    "title": "RAMP-CNN: A Novel Neural Network for Enhanced Automotive Radar Object Recognition",
    "publish_time": "20201113",
    "authors": [
      "Xiangyu Gao",
      "Guanbin Xing",
      "Sumit Roy",
      "and Hui Liu"
    ],
    "abstract": "Millimeter-wave radars are being increasingly integrated into commercial vehicles to support new advanced driver-assistance systems by enabling robust and high-performance object detection, localization, as well as recognition - a key component of new environmental perception. In this paper, we propose a novel radar multiple-perspectives convolutional neural network (RAMP-CNN) that extracts the location and class of objects based on further processing of the range-velocity-angle (RVA) heatmap sequences. To bypass the complexity of 4D convolutional neural networks (NN), we propose to combine several lower-dimension NN models within our RAMP-CNN model that nonetheless approaches the performance upper-bound with lower complexity. The extensive experiments show that the proposed RAMP-CNN model achieves better average recall and average precision than prior works in all testing scenarios. Besides, the RAMP-CNN model is validated to work robustly under nighttime, which enables low-cost radars as a potential substitute for pure optical sensing under severe conditions.",
    "score": 0.5921449065208435
  },
  {
    "link": "https://www.arxiv.org/abs/2209.14499",
    "title": "NVRadarNet: Real-Time Radar Obstacle and Free Space Detection for Autonomous Driving",
    "publish_time": "20220929",
    "authors": [
      "Alexander Popov",
      "Patrik Gebhardt",
      "Ke Chen",
      "Ryan Oldja",
      "Heeseok Lee",
      "Shane Murray",
      "Ruchi Bhargava",
      "Nikolai Smolyanskiy"
    ],
    "abstract": "Detecting obstacles is crucial for safe and efficient autonomous driving. To this end, we present NVRadarNet, a deep neural network (DNN) that detects dynamic obstacles and drivable free space using automotive RADAR sensors. The network utilizes temporally accumulated data from multiple RADAR sensors to detect dynamic obstacles and compute their orientation in a top-down bird's-eye view (BEV). The network also regresses drivable free space to detect unclassified obstacles. Our DNN is the first of its kind to utilize sparse RADAR signals in order to perform obstacle and free space detection in real time from RADAR data only. The network has been successfully used for perception on our autonomous vehicles in real self-driving scenarios. The network runs faster than real time on an embedded GPU and shows good generalization across geographic regions.",
    "score": 0.49961668252944946
  }
]
```