## Synomaly Noise and Multi-Stage Diffusion: A Novel Approach for Unsupervised Anomaly Detection in Medical Images

Anomaly detection in medical imaging plays a crucial role in identifying pathological regions across various imaging modalities, such as brain MRI, liver CT, and carotid ultrasound (US). However, training fully supervised segmentation models is often hindered by the scarcity of expert annotations and the complexity of diverse anatomical structures. To address these issues, we propose a novel unsupervised anomaly detection framework based on a diffusion model that incorporates a synthetic anomaly (Synomaly) noise function and a multi-stage diffusion process. Synomaly noise introduces synthetic anomalies into healthy images during training, allowing the model to effectively learn anomaly removal. The multi-stage diffusion process is introduced to progressively denoise images, preserving fine details while improving the quality of anomaly-free reconstructions. The generated high-fidelity counterfactual healthy images can further enhance the interpretability of the segmentation models, as well as provide a reliable baseline for evaluating the extent of anomalies and supporting clinical decision-making. Notably, the unsupervised anomaly detection model is trained purely on healthy images, eliminating the need for anomalous training samples and pixel-level annotations.

<div align="center">
<img src=images/Overview.png  width=80%/>
</div>
Overview of the proposed approach.(a) illustrates the training process of the proposed unsupervised anomaly detection approach, where Synomaly noise is utilized to corrupt the healthy image. (b) depicts the inference processs, in which Gaussian noise is applied directly to the anomalous images. Since the trained model has already been exposed to similar synthetic anomalies during training, it can efficiently erase the abnormalities. Multi-stage diffusion process is implemented to conserve fine details of the original image, thus increasing the anomaly detection accuracy.

<div align="center">
<img src=images/Overview.png  width=80%/>
</div>


The pretrained model can be accessed via this link: https://drive.google.com/file/d/1j6PZBbQA5PEUmhjrzixC4kDYkpdlO2p0/view?usp=sharing
