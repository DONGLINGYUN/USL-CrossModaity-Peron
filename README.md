# USL Cross-Modality Person RE-identification
#### We introduced a semi-supervised learning method in cross-modal person re-identification, whose main purpose is to generate high-quality cross-modal image pairs guided by confidence scores based on supervised information, and update the memory bank. By using the interaction between features of different modalities, the model achieves the effect of reducing modal differences. Our model mainly includes two modules: Confidence Guided cross-modality pseudo-label generation and semi-supervised cross-modalality discriminative feature learning. 

# Framework
![fig.JPEG](./1.JPG)
=======================
# Requirement
Dependency packages that the environment may require：
[requirement.txt](./requirement.txt)

# Train
`<1. sh run_train_sysu.sh for SYSU-MM01>`  
`<2. sh run_train_regdb.sh for RegD >`  

`<hello world>`  

# Test 
`<1. sh run_test_sysu.sh for SYSU-MM01>`  
`<2. sh run_test_regdb.sh for RegDB>`  

# Notice 
When we use the RegDB dataset for training or testing, we need to switch to a different branch network when query is visible or infrared. The Settings on Gallary are similar. We have some trained models available on ！[Badi du disk](https://pan.baidu.com/s/12_Kudw-4WveWr7XjpuUNkA ),code：REID. And more versions will be updated gradually,
