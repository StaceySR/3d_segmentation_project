# 3d_interactive_segmentation
## Overview
This project is about doing segmentation on 3d medical images with an extra input from users' clicks on the objects they want to segment. The coding of this project is based on Samsung AI Center's 2020 f-brs click-based interactive segementation project (to access their github project, please use the link in References section). In that project, the click-based interactive segmentation is done on RGB 2d pictures, and the main accomplishment of this project is extending the application to 3d volumetric medical CT images. The architecture of the deep neural network for this project is deeplab v3+ with 3D Resnet-34 backbones. To demonstrate the interactive segmentation process, I also modify the demo app from the original 2d project and make it able to show and segment 3d images page by page. 
## Training
Training of this project is done on Google Colab. The datasets that we choose for this project are spleen, liver, and lung datasets from <a href=http://medicaldecathlon.com/ title="Flaticon">medical dedcathlon</a>. 
## Evaluation and Demo
## References
* This project is based on the <a href="https://github.com/saic-vul/fbrs_interactive_segmentation" title="Flaticon">f-brs click-based interactive segementation project</a>, and therefore should be as a reference to that project. 
