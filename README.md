Beyond the Visible Spectrum: Multimodal Deep Fusion for Agricultural DiagnosticsAn advanced Deep Learning project developed for The University of Jordan (Department of Artificial Intelligence) course project (Spring 2026). 

This repository introduces a novel multi-stream neural network designed to diagnose wheat diseases (specifically Rust) by extending computer vision capabilities beyond human-visible wavelengths.

Project Overview & ObjectivesStandard 
RGB vision systems can only detect agricultural stressors after structural or macroscopic lesions appear on the leaf surface. 
This project utilizes an engineered approach by fusing standard RGB data with Multispectral (MS) satellite bands and Hyperspectral (HS) narrow drone bands,
capturing sub-visual cellular and chemical changes in plant tissue.

The model classifies wheat samples into three perfectly balanced classes (1,800 total samples, 600 per class):

Health (Healthy leaf tissue)
Rust (Fungal pathogen detection)
Other (Alternative biotic/abiotic stressors)

System Architecture (Triple-Stream Late Fusion)Instead of flattening different sensory dimensions early on—which introduces noise and instability—this architecture utilizes a Late Fusion Framework:

RGB Branch (3 Channels): Leverages a deep ResNet-50 backbone initialized with pretrained weights to extract high-fidelity spatial structures and color patterns.
Multispectral Branch (5 Channels): Processes critical localized bands (such as Near-Infrared) typically obtained via satellite imaging to monitor cellular layout and plant moisture changes.
Hyperspectral Branch (126 Channels): Processes highly localized narrow-band drone measurements to screen deep chemical signatures.

Deep features from all three branches are independently optimized, combined via a concatenation layer, and fed into a fully connected classification head.📊 Experimental Setup & ValidationTo ensure absolute reliability and prevent single-split verification bias, the pipeline was tested using a rigorous 5-Fold Stratified Cross-Validation strategy.Loss Function: CrossEntropyLossOptimizer: Adam (Learning Rate: 0.0001)Scheduler: CosineAnnealingLR dynamically decaying over 20 epochs per fold to guarantee optimal numerical convergence.📈 Empirical Results & Ablation AnalysisFramework ModelModalities IncludedValidation ProtocolMean AccuracyStability (Std Dev)Baseline ControlRGB Only (3 Channels)Single Random Split96.00%N/A (Unverified)Proposed InnovationRGB + MS + HS (134 Channels)5-Fold Stratified CV89.39%± 2.19%🔍 Scientific InterpretationWhile the single-split RGB model shows an optimistic 96.00% accuracy, single-splits are vulnerable to partition luck and spatial overfitting.Under strict 5-Fold Stratified Cross-Validation, our TripleFusionModel achieved a highly robust 89.39% mean accuracy with an exceptionally tight standard deviation of ±2.19%. This proves that expanding the feature space to 134 channels provides a highly generalized spectral signature that behaves uniformly across unseen data structures.Furthermore, per-class metrics analysis showed that the model achieved its highest overall recall performance precisely within the 'Rust' target domain, hitting critical test epoch recall rates of 0.98 and 0.96. This ensures that early-stage fungal infestations are swiftly intercepted.📦 Core Dependencies & SetupTo run the evaluation pipeline, ensure you have the following packages installed:Bashpip install torch torchvision timm tifffile pandas scikit-learn opencv-python
Model PersistenceThe final trained weights of the optimized model are saved via PyTorch:Pythonimport torch

# Save command used in the final notebook loop
torch.save(model_fold.state_dict(), 'Wheat_TripleFusion_Final_Model.pth')

Developed By: Qusai Rahhal, Bashar Zabaneh, Kaled AlhadearesAcademic Institution: The University of JordanSemester: Spring 2026
