[PaSa](https://pasa-agent.ai/home?query=papers+about+3d+segmentation+with+radar+in+autonomous+driving&session=1741178918686373868)

```json
[
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
    "score": 0.9803035259246826
  },
  {
    "link": "https://www.arxiv.org/abs/2004.03451",
    "title": "RSS-Net: Weakly-Supervised Multi-Class Semantic Segmentation with FMCW Radar",
    "publish_time": "20200402",
    "authors": [
      "Prannay Kaul",
      "Daniele De Martini",
      "Matthew Gadd",
      "Paul Newman"
    ],
    "abstract": "This paper presents an efficient annotation procedure and an application thereof to end-to-end, rich semantic segmentation of the sensed environment using FMCW scanning radar. We advocate radar over the traditional sensors used for this task as it operates at longer ranges and is substantially more robust to adverse weather and illumination conditions. We avoid laborious manual labelling by exploiting the largest radar-focused urban autonomy dataset collected to date, correlating radar scans with RGB cameras and LiDAR sensors, for which semantic segmentation is an already consolidated procedure. The training procedure leverages a state-of-the-art natural image segmentation system which is publicly available and as such, in contrast to previous approaches, allows for the production of copious labels for the radar stream by incorporating four camera and two LiDAR streams. Additionally, the losses are computed taking into account labels to the radar sensor horizon by accumulating LiDAR returns along a pose-chain ahead and behind of the current vehicle position. Finally, we present the network with multi-channel radar scan inputs in order to deal with ephemeral and dynamic scene objects.",
    "score": 0.956598699092865
  },
  {
    "link": "https://www.arxiv.org/abs/2409.04979",
    "title": "RCBEVDet++: Toward High-accuracy Radar-Camera Fusion 3D Perception Network",
    "publish_time": "20240908",
    "authors": [
      "Zhiwei Lin",
      "Zhe Liu",
      "Yongtao Wang",
      "Le Zhang",
      "Ce Zhu"
    ],
    "abstract": "Perceiving the surrounding environment is a fundamental task in autonomous driving. To obtain highly accurate perception results, modern autonomous driving systems typically employ multi-modal sensors to collect comprehensive environmental data. Among these, the radar-camera multi-modal perception system is especially favored for its excellent sensing capabilities and cost-effectiveness. However, the substantial modality differences between radar and camera sensors pose challenges in fusing information. To address this problem, this paper presents RCBEVDet, a radar-camera fusion 3D object detection framework. Specifically, RCBEVDet is developed from an existing camera-based 3D object detector, supplemented by a specially designed radar feature extractor, RadarBEVNet, and a Cross-Attention Multi-layer Fusion (CAMF) module. Firstly, RadarBEVNet encodes sparse radar points into a dense bird's-eye-view (BEV) feature using a dual-stream radar backbone and a Radar Cross Section aware BEV encoder. Secondly, the CAMF module utilizes a deformable attention mechanism to align radar and camera BEV features and adopts channel and spatial fusion layers to fuse them. To further enhance RCBEVDet's capabilities, we introduce RCBEVDet++, which advances the CAMF through sparse fusion, supports query-based multi-view camera perception models, and adapts to a broader range of perception tasks. Extensive experiments on the nuScenes show that our method integrates seamlessly with existing camera-based 3D perception models and improves their performance across various perception tasks. Furthermore, our method achieves state-of-the-art radar-camera fusion results in 3D object detection, BEV semantic segmentation, and 3D multi-object tracking tasks. Notably, with ViT-L as the image backbone, RCBEVDet++ achieves 72.73 NDS and 67.34 mAP in 3D object detection without test-time augmentation or model ensembling.",
    "score": 0.9509229063987732
  },
  {
    "link": "https://www.arxiv.org/abs/2103.16214",
    "title": "Multi-View Radar Semantic Segmentation",
    "publish_time": "20210330",
    "authors": [
      "Arthur Ouaknine",
      "Alasdair Newson",
      "Patrick P\\'erez",
      "Florence Tupin",
      "Julien Rebut"
    ],
    "abstract": "Understanding the scene around the ego-vehicle is key to assisted and autonomous driving. Nowadays, this is mostly conducted using cameras and laser scanners, despite their reduced performances in adverse weather conditions. Automotive radars are low-cost active sensors that measure properties of surrounding objects, including their relative speed, and have the key advantage of not being impacted by rain, snow or fog. However, they are seldom used for scene understanding due to the size and complexity of radar raw data and the lack of annotated datasets. Fortunately, recent open-sourced datasets have opened up research on classification, object detection and semantic segmentation with raw radar signals using end-to-end trainable models. In this work, we propose several novel architectures, and their associated losses, which analyse multiple \"views\" of the range-angle-Doppler radar tensor to segment it semantically. Experiments conducted on the recent CARRADA dataset demonstrate that our best model outperforms alternative models, derived either from the semantic segmentation of natural images or from radar scene understanding, while requiring significantly fewer parameters. Both our code and trained models are available at https://github.com/valeoai/MVRSS.",
    "score": 0.9451363682746887
  },
  {
    "link": "https://www.arxiv.org/abs/2405.14014",
    "title": "RadarOcc: Robust 3D Occupancy Prediction with 4D Imaging Radar",
    "publish_time": "20240522",
    "authors": [
      "Fangqiang Ding",
      "Xiangyu Wen",
      "Yunzhou Zhu",
      "Yiming Li",
      "Chris Xiaoxuan Lu"
    ],
    "abstract": "3D occupancy-based perception pipeline has significantly advanced autonomous driving by capturing detailed scene descriptions and demonstrating strong generalizability across various object categories and shapes. Current methods predominantly rely on LiDAR or camera inputs for 3D occupancy prediction. These methods are susceptible to adverse weather conditions, limiting the all-weather deployment of self-driving cars. To improve perception robustness, we leverage the recent advances in automotive radars and introduce a novel approach that utilizes 4D imaging radar sensors for 3D occupancy prediction. Our method, RadarOcc, circumvents the limitations of sparse radar point clouds by directly processing the 4D radar tensor, thus preserving essential scene details. RadarOcc innovatively addresses the challenges associated with the voluminous and noisy 4D radar data by employing Doppler bins descriptors, sidelobe-aware spatial sparsification, and range-wise self-attention mechanisms. To minimize the interpolation errors associated with direct coordinate transformations, we also devise a spherical-based feature encoding followed by spherical-to-Cartesian feature aggregation. We benchmark various baseline methods based on distinct modalities on the public K-Radar dataset. The results demonstrate RadarOcc's state-of-the-art performance in radar-based 3D occupancy prediction and promising results even when compared with LiDAR- or camera-based methods. Additionally, we present qualitative evidence of the superior performance of 4D radar in adverse weather conditions and explore the impact of key pipeline components through ablation studies.",
    "score": 0.7764768004417419
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
    "score": 0.776387095451355
  },
  {
    "link": "https://www.arxiv.org/abs/2103.03387",
    "title": "PolarNet: Accelerated Deep Open Space Segmentation Using Automotive Radar in Polar Domain",
    "publish_time": "20210304",
    "authors": [
      "Farzan Erlik Nowruzi",
      "Dhanvin Kolhatkar",
      "Prince Kapoor",
      "Elnaz Jahani Heravi",
      "Fahed Al Hassanat",
      "Robert Laganiere",
      "Julien Rebut",
      "Waqas Malik"
    ],
    "abstract": "Camera and Lidar processing have been revolutionized with the rapid development of deep learning model architectures. Automotive radar is one of the crucial elements of automated driver assistance and autonomous driving systems. Radar still relies on traditional signal processing techniques, unlike camera and Lidar based methods. We believe this is the missing link to achieve the most robust perception system. Identifying drivable space and occupied space is the first step in any autonomous decision making task. Occupancy grid map representation of the environment is often used for this purpose. In this paper, we propose PolarNet, a deep neural model to process radar information in polar domain for open space segmentation. We explore various input-output representations. Our experiments show that PolarNet is a effective way to process radar data that achieves state-of-the-art performance and processing speeds while maintaining a compact size.",
    "score": 0.7763601541519165
  },
  {
    "link": "https://www.arxiv.org/abs/2412.10734",
    "title": "OmniHD-Scenes: A Next-Generation Multimodal Dataset for Autonomous Driving",
    "publish_time": "20241214",
    "authors": [
      "Lianqing Zheng",
      "Long Yang",
      "Qunshu Lin",
      "Wenjin Ai",
      "Minghao Liu",
      "Shouyi Lu",
      "Jianan Liu",
      "Hongze Ren",
      "Jingyue Mo",
      "Xiaokai Bai",
      "Jie Bai",
      "Zhixiong Ma",
      "Xichan Zhu"
    ],
    "abstract": "The rapid advancement of deep learning has intensified the need for comprehensive data for use by autonomous driving algorithms. High-quality datasets are crucial for the development of effective data-driven autonomous driving solutions. Next-generation autonomous driving datasets must be multimodal, incorporating data from advanced sensors that feature extensive data coverage, detailed annotations, and diverse scene representation. To address this need, we present OmniHD-Scenes, a large-scale multimodal dataset that provides comprehensive omnidirectional high-definition data. The OmniHD-Scenes dataset combines data from 128-beam LiDAR, six cameras, and six 4D imaging radar systems to achieve full environmental perception. The dataset comprises 1501 clips, each approximately 30-s long, totaling more than 450K synchronized frames and more than 5.85 million synchronized sensor data points. We also propose a novel 4D annotation pipeline. To date, we have annotated 200 clips with more than 514K precise 3D bounding boxes. These clips also include semantic segmentation annotations for static scene elements. Additionally, we introduce a novel automated pipeline for generation of the dense occupancy ground truth, which effectively leverages information from non-key frames. Alongside the proposed dataset, we establish comprehensive evaluation metrics, baseline models, and benchmarks for 3D detection and semantic occupancy prediction. These benchmarks utilize surround-view cameras and 4D imaging radar to explore cost-effective sensor solutions for autonomous driving applications. Extensive experiments demonstrate the effectiveness of our low-cost sensor configuration and its robustness under adverse conditions. Data will be released at https://www.2077ai.com/OmniHD-Scenes.",
    "score": 0.7539199590682983
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
    "score": 0.7049328088760376
  },
  {
    "link": "https://www.arxiv.org/abs/2501.15394",
    "title": "Doracamom: Joint 3D Detection and Occupancy Prediction with Multi-view 4D Radars and Cameras for Omnidirectional Perception",
    "publish_time": "20250126",
    "authors": [
      "Lianqing Zheng",
      "Jianan Liu",
      "Runwei Guan",
      "Long Yang",
      "Shouyi Lu",
      "Yuanzhe Li",
      "Xiaokai Bai",
      "Jie Bai",
      "Zhixiong Ma",
      "Hui-Liang Shen",
      "and Xichan Zhu"
    ],
    "abstract": "3D object detection and occupancy prediction are critical tasks in autonomous driving, attracting significant attention. Despite the potential of recent vision-based methods, they encounter challenges under adverse conditions. Thus, integrating cameras with next-generation 4D imaging radar to achieve unified multi-task perception is highly significant, though research in this domain remains limited. In this paper, we propose Doracamom, the first framework that fuses multi-view cameras and 4D radar for joint 3D object detection and semantic occupancy prediction, enabling comprehensive environmental perception. Specifically, we introduce a novel Coarse Voxel Queries Generator that integrates geometric priors from 4D radar with semantic features from images to initialize voxel queries, establishing a robust foundation for subsequent Transformer-based refinement. To leverage temporal information, we design a Dual-Branch Temporal Encoder that processes multi-modal temporal features in parallel across BEV and voxel spaces, enabling comprehensive spatio-temporal representation learning. Furthermore, we propose a Cross-Modal BEV-Voxel Fusion module that adaptively fuses complementary features through attention mechanisms while employing auxiliary tasks to enhance feature quality. Extensive experiments on the OmniHD-Scenes, View-of-Delft (VoD), and TJ4DRadSet datasets demonstrate that Doracamom achieves state-of-the-art performance in both tasks, establishing new benchmarks for multi-modal 3D perception. Code and models will be publicly available.",
    "score": 0.7048570513725281
  },
  {
    "link": "https://www.arxiv.org/abs/2005.01456",
    "title": "CARRADA Dataset: Camera and Automotive Radar with Range-Angle-Doppler Annotations",
    "publish_time": "20200504",
    "authors": [
      "A. Ouaknine",
      "A. Newson",
      "J. Rebut",
      "F. Tupin and P. P\\'erez"
    ],
    "abstract": "High quality perception is essential for autonomous driving (AD) systems. To reach the accuracy and robustness that are required by such systems, several types of sensors must be combined. Currently, mostly cameras and laser scanners (lidar) are deployed to build a representation of the world around the vehicle. While radar sensors have been used for a long time in the automotive industry, they are still under-used for AD despite their appealing characteristics (notably, their ability to measure the relative speed of obstacles and to operate even in adverse weather conditions). To a large extent, this situation is due to the relative lack of automotive datasets with real radar signals that are both raw and annotated. In this work, we introduce CARRADA, a dataset of synchronized camera and radar recordings with range-angle-Doppler annotations. We also present a semi-automatic annotation approach, which was used to annotate the dataset, and a radar semantic segmentation baseline, which we evaluate on several metrics. Both our code and dataset are available online.",
    "score": 0.6506100296974182
  },
  {
    "link": "https://www.arxiv.org/abs/2203.01137",
    "title": "Self-Supervised Scene Flow Estimation with 4-D Automotive Radar",
    "publish_time": "20220302",
    "authors": [
      "Fangqiang Ding",
      "Zhijun Pan",
      "Yimin Deng",
      "Jianning Deng",
      "Chris Xiaoxuan Lu"
    ],
    "abstract": "Scene flow allows autonomous vehicles to reason about the arbitrary motion of multiple independent objects which is the key to long-term mobile autonomy. While estimating the scene flow from LiDAR has progressed recently, it remains largely unknown how to estimate the scene flow from a 4-D radar - an increasingly popular automotive sensor for its robustness against adverse weather and lighting conditions. Compared with the LiDAR point clouds, radar data are drastically sparser, noisier and in much lower resolution. Annotated datasets for radar scene flow are also in absence and costly to acquire in the real world. These factors jointly pose the radar scene flow estimation as a challenging problem. This work aims to address the above challenges and estimate scene flow from 4-D radar point clouds by leveraging self-supervised learning. A robust scene flow estimation architecture and three novel losses are bespoken designed to cope with intractable radar data. Real-world experimental results validate that our method is able to robustly estimate the radar scene flow in the wild and effectively supports the downstream task of motion segmentation.",
    "score": 0.6218354105949402
  },
  {
    "link": "https://www.arxiv.org/abs/2110.01775",
    "title": "Deep Instance Segmentation with Automotive Radar Detection Points",
    "publish_time": "20211005",
    "authors": [
      "Jianan Liu",
      "Weiyi Xiong",
      "Liping Bai",
      "Yuxuan Xia",
      "Tao Huang",
      "Wanli Ouyang",
      "Bing Zhu"
    ],
    "abstract": "Automotive radar provides reliable environmental perception in all-weather conditions with affordable cost, but it hardly supplies semantic and geometry information due to the sparsity of radar detection points. With the development of automotive radar technologies in recent years, instance segmentation becomes possible by using automotive radar. Its data contain contexts such as radar cross section and micro-Doppler effects, and sometimes can provide detection when the field of view is obscured. The outcome from instance segmentation could be potentially used as the input of trackers for tracking targets. The existing methods often utilize a clustering-based classification framework, which fits the need of real-time processing but has limited performance due to minimum information provided by sparse radar detection points. In this paper, we propose an efficient method based on clustering of estimated semantic information to achieve instance segmentation for the sparse radar detection points. In addition, we show that the performance of the proposed approach can be further enhanced by incorporating the visual multi-layer perceptron. The effectiveness of the proposed method is verified by experimental results on the popular RadarScenes dataset, achieving 89.53% mean coverage and 86.97% mean average precision with the IoU threshold of 0.5, which is superior to other approaches in the literature. More significantly, the consumed memory is around 1MB, and the inference time is less than 40ms, indicating that our proposed algorithm is storage and time efficient. These two criteria ensure the practicality of the proposed method in real-world systems.",
    "score": 0.5614910125732422
  },
  {
    "link": "https://www.arxiv.org/abs/2411.15016",
    "title": "MSSF: A 4D Radar and Camera Fusion Framework With Multi-Stage Sampling for 3D Object Detection in Autonomous Driving",
    "publish_time": "20241122",
    "authors": [
      "Hongsi Liu",
      "Jun Liu",
      "Guangfeng Jiang",
      "Xin Jin"
    ],
    "abstract": "As one of the automotive sensors that have emerged in recent years, 4D millimeter-wave radar has a higher resolution than conventional 3D radar and provides precise elevation measurements. But its point clouds are still sparse and noisy, making it challenging to meet the requirements of autonomous driving. Camera, as another commonly used sensor, can capture rich semantic information. As a result, the fusion of 4D radar and camera can provide an affordable and robust perception solution for autonomous driving systems. However, previous radar-camera fusion methods have not yet been thoroughly investigated, resulting in a large performance gap compared to LiDAR-based methods. Specifically, they ignore the feature-blurring problem and do not deeply interact with image semantic information. To this end, we present a simple but effective multi-stage sampling fusion (MSSF) network based on 4D radar and camera. On the one hand, we design a fusion block that can deeply interact point cloud features with image features, and can be applied to commonly used single-modal backbones in a plug-and-play manner. The fusion block encompasses two types, namely, simple feature fusion (SFF) and multiscale deformable feature fusion (MSDFF). The SFF is easy to implement, while the MSDFF has stronger fusion abilities. On the other hand, we propose a semantic-guided head to perform foreground-background segmentation on voxels with voxel feature re-weighting, further alleviating the problem of feature blurring. Extensive experiments on the View-of-Delft (VoD) and TJ4DRadset datasets demonstrate the effectiveness of our MSSF. Notably, compared to state-of-the-art methods, MSSF achieves a 7.0% and 4.0% improvement in 3D mean average precision on the VoD and TJ4DRadSet datasets, respectively. It even surpasses classical LiDAR-based methods on the VoD dataset.",
    "score": 0.4994848966598511
  },
  {
    "link": "https://www.arxiv.org/abs/1904.00415",
    "title": "Road Scene Understanding by Occupancy Grid Learning from Sparse Radar Clusters using Semantic Segmentation",
    "publish_time": "20190331",
    "authors": [
      "Liat Sless",
      "Gilad Cohen",
      "Bat El Shlomo",
      "Shaul Oron"
    ],
    "abstract": "Occupancy grid mapping is an important component in road scene understanding for autonomous driving. It encapsulates information of the drivable area, road obstacles and enables safe autonomous driving. Radars are an emerging sensor in autonomous vehicle vision, becoming more widely used due to their long range sensing, low cost, and robustness to severe weather conditions. Despite recent advances in deep learning technology, occupancy grid mapping from radar data is still mostly done using classical filtering approaches.In this work, we propose learning the inverse sensor model used for occupancy grid mapping from clustered radar data. This is done in a data driven approach that leverages computer vision techniques. This task is very challenging due to data sparsity and noise characteristics of the radar sensor. The problem is formulated as a semantic segmentation task and we show how it can be learned using lidar data for generating ground truth. We show both qualitatively and quantitatively that our learned occupancy net outperforms classic methods by a large margin using the recently released NuScenes real-world driving data.",
    "score": 0.4374634325504303
  }
]
```

