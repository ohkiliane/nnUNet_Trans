# nnUNet_Trans
This ensemble approach is developed for brain tumor segmentation.
The paper 'Ensemble Learning with Residual Transformer for Brain Tumor Segmentation' is accepted for ISBI. (link: TBD)

Brain tumor segmentation is an active research area due to the difficulty in delineating highly complex shaped and textured tumors as well as the failure of the commonly used U-Net architectures in such. The combination of different neural architectures is among the mainstream research recently, particularly the combination of U-Net with Transformers because of their innate attention mechanism and pixel-wise labeling. Different from previous efforts, this paper proposed a novel network architecture that integrates Transformers into a self-adaptive U-Net to draw out 3D volumetric contexts with reasonable computational costs. We further added residual connection to prevent degradation in information flow and explored ensemble methods, as the evaluated models have edges on different cases and sub-regions. On the BraTS 2021 dataset (3D), our models outperformed state-of-the-art segmentation methods. Our findings may direct future research on potential ways of combining multiple architectures and their fusions for optimal segmentation of brain tumors.

Four models are 
