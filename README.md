# 3d_interactive_segmentation
## Overview
This project is about doing segmentation on 3d medical images with an extra input from users' clicks on the objects they want to segment. The coding of this project is done by modifying codes from Samsung AI Center's 2020 f-brs click-based interactive segementation project. In that project, the click-based interactive segmentation is done on RGB 2d pictures, and the main accomplishment of this project is extending the application to 3d volumetric medical CT images. The architecture of the deep neural network for this project is deeplab v3+ with 3D Resnet-34 backbones. To demonstrate the interactive segmentation process, I also modify the demo app from the original 2d project and make it able to show and segment 3d images page by page. 
## Training
Training of this project is done on Google Colab. The datasets that we choose for this project are spleen, liver, and lung datasets from <a href=http://medicaldecathlon.com/ title="Flaticon">medical decathlon</a>. To train datasets with this project, please first download datasets from medical decathlon official website, create a new empty directory "datasets", and put downdloaded datasets into this directory. Then include the path in 
## Evaluation and Demo
![](demo.gif)
## License Notice and References
* This project is based on the <a href="https://github.com/saic-vul/fbrs_interactive_segmentation" title="Flaticon">f-brs click-based interactive segementation project</a>, and therefore should be as a reference to that project. Since the original project has MPL 2.0 License, all files in this project are also licensed under this license.
* APA format of their paper reference: Sofiiuk, K., Petrov, I., Barinova, O., Konushin A. (2020)  Rethinking Backpropagating Refinement for Interactive Segmentation, arXiv preprint arXiv:2001.10331
