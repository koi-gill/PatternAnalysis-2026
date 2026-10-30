# Documentation via README.md - 10 Marks
Once your algorithm has been implemented, you will need to provide sufficient comments in your
code and comprehensive documentation of your algorithm in a README.md file. The readme file should
contain:
- A title, a description of the algorithm and the problem that it solves (approximately a paragraph),
how it works in a paragraph, and a figure/visualisation.
- Your one-page written feasibility review from Section 3.
- Any dependencies required, including exact versions, and address reproducibility of results.
- Example inputs, outputs, and visual plots of your algorithm.
- Description of any specific pre-processing used, with references if any. Justify your training,
validation, and testing splits of the data.
- Investigation of the Open Research Dilemma:
- Quantitative benchmarking of your chosen model against your implemented baseline model under
identical evaluation splits.
14
- Resource profiling table documenting computational efficiency (peak GPU VRAM, parameter
count, and inference runtime/latency).
- In-depth qualitative error autopsies analyzing 3–5 representative failure cases from the dataset
(diagnosing root triggers and failure modes).
- Actionable engineering recommendation and trade-off synthesis for your project manager.
- The mandatory Artificial Intelligence Usage Disclosure subsection (see Section 6).
Marking Criteria:
1. Description and explanation of the working principles of the algorithm implemented and the problem
it solves (5 Marks).
2. Description of usage and comments throughout scripts (3 marks).
3. Proper formatting using GitHub markdown (2 marks).

---

# 2D vs 3D U-Net for Prostate Segmentation in Radiotherapy MRI (HipMRI)

## 1. Algorithm and Problem Description
External-beam radiotherapy for prostate cancer requires the prostate to be contoured on all MRI slices with millimeter precision. Under-contouring poses a risk of tumor reoccurence, and over-contouring can irradiate neighboring organs and tissue potentially causing inflammation and damage. Manually contouring is a slow and tedious process, hence, this project tests a deep-learning drafting assistant that proposes contours for a specialist to review. The investigation is the dilemma of spatial context in this model. A 2D model segments each slice on its own and is relatively cheap to train, while a 3D model can use the spatial context of the whole 3D volume and in theory should produce more continuous contours. 

![Original U-Net architecture: a contracting encoder (64 to 1024 channels) and an expanding decoder joined by copy-and-crop skip connections, producing a 2-class segmentation map.](images/Screenshot%202026-10-30%20123109.png)
*Figure 1: Basic U-Net architecture as provided in the lecture slides*

![MRI U-Net: a 1-channel input, a four-level encoder (16 to 128 channels), a 256-channel bottleneck, and a decoder with skip connections, ending in a 1×1 conv to 6 class logits](images/UNet%20architecture.drawio.png)
*Figure 2: U-Net architecture used for both the 2D and 3D U-Net models. Only difference is the convolution dimensions*



# 2. Feasibility Review
## User Need, Focus, and Acceptance Criteria
User is a radiotherapy imaging specialist that works with anatomical contours of the prostate gland
Prototype purpose: to investigate whether the true 3d volumetric continuity of a 3D U-Net model can eliminate the boundary and slice-transition errors of a 2D U-Net baseline in prostate radiotherapy planning, or if it only refines the easy central slices while consuming much more compute.
## Acceptance criteria
- Prostate label dice sim. >= 0.75 for 2d U-Net, global dice sim. >= 0.7 for 3d U-Net (test patients)
    - Global dice similarity may not capture the success of the model for the base and apex slices, but provides a good validity check
- Stratified Dice similarity by slice position (base, mid, apex) accompanied by HD95 to quantify the gap between predicted boundary and ground truth for each stratum.
    - This helps determine and quantify where 3D U-Net differs from 2D U-Net for each primary area of the prostate scans.
- Peak GPU VRAM and inference time per volume stays under 4 GB/5 seconds for 2D U-Net and 12 GB/15 seconds for 3D U-Net
## Model Choice & Course Concepts
- Dataset configuration:
    - HipMRI study: 38 patients, 211 3d volumes, labels: body, bone, bladder, rectum, prostate. 
    - Patient-level splitting: all slices from a given patient end up in the same train/validation/test split. Prevents leakage between sets. typical split ratio to be ## used: 70/15/15 -> 26/6/6 patients. 
- Preprocessing: 
    - Normalise image intensity per image using z-score; One-hot encoding for labels
- Model architecture:
    - 4 levels (3 downsamples)
    - Filters starting at 32/16, doubling each level.
    - Block: 2 x (3x3 conv, norm, ReLU) (3d convolution for 3D UNet)
    - Models kept similar in architecture aside from increase in convolution dimension – this should allow test results to better reflect the effectiveness of using a 3D UNet over a 2D one. 
## Preliminary Feasibility Evidence:
Initial Data Audit: scans are of 38 patients some over a period of up to 7 weeks of radiotherapy – for the 3d data it is difficult to tell what scans belong to a single patient.
Results of baseline smoke-test experiment/resource estimate for TinyUNet3d on cloud-hosted GPU: 
- (64, 64, 32) peak VRAM: 0.07 GiB
- (128, 128, 64) peak VRAM: 0.56 GiB
## Risks, Budget, and Fallback
Training 3D-UNet may exceed memory or time limitations – mitigation: train using sub-sections to begin with; mixed precision training, possibly lower input resolution if needed
Only 38 patients – stratified results may be noisy, and test set only contains 6 patients with the proposed splitting ratio. Treatment: report per-patient values for test set. Could possibly implement k-fold cross val
Primary computing budget: using the cloud hosted GPUs run by UQ’s GPU lab – hard quota of 26 hrs run time total.
Fallback – code should be compatible with Rangpur so that if GPU lab quota run’s out, it can be run on the Rangpur GPUs
2D Improved UNet as a back up model

# 3. Data, Pre-processing and Splits

# 4. Training Setup

# 5. Results: Investigating the Spatial Context Dilemma

# 6. Recommendations for the Project Manager

# 7. Usage

# 8. Dependencies and Reproducibility

# 9. Artificial Intelligence Disclosure