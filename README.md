<h1 align="center">A graph-based multi-modal data fusion framework for identifying urban functional zone</h1>


![](images/Flowchart.jpg)
    
This is the official PyTorch implementation of the paper **[A graph-based multi-modal data fusion framework for identifying urban functional zone]()**.

Note: The app and codes we developed, data, and trained models will be released upon acceptance of the paper.

GUI of **An Intelligent Toolbox for Identifying Urban Functional Zone**, which is developed based on the proposed G2MF.

 ![](https://youtu.be/FZLEjUOS45I)
[![Watch the video](https://www.youtube.com/watch?v=FZLEjUOS45I/maxresdefault.jpg)](https://www.youtube.com/watch?v=FZLEjUOS45I)

<video controls>
  <source src="images/GUI.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Geographical location of study areas
  
  ![](images/Location.jpg)


### Table of content
1. [Preparation](#preparation)
2. [Run APP](#APP)
3. [Training G2MF](#G2MF)
4. [UFZ Identification](#results)
5. [Sensitivity Analysis of G2MF](#discussion)
6. [Paper](#paper)
7. [Outlook](#outlook)
8. [License](#license)

### 1. Preparation
- Recommendation: Install a virtual env: `conda create -n g2mf python=3.11`
- Install required packages: `pip install -r requirements.txt`
  
### 2. Run APP <a name="APP"></a>
- Run APP:
```
$ cd G2MF-main/app
$ python run.py
```

### 3. Training G2MF <a name="G2MF"></a>
- Train G2MF:
```
$ cd G2MF-main/src
$ python train.py --data_dir /Path/To/DATASET/
```

### 4. UFZ Identification <a name="results"></a>
- **Beijing-Shanghai:** Visualization of some UFZ inference results in Beijing and Shanghai. We show some typical places, such as the university campus in Haidian District (Beijing), green spaces and squares surrounding the Forbidden City (Beijing), the Tiantongyuan residential area in Changping District (Beijing), the university campus in Yangpu District (Shanghai), the Lujiazui commercial area in Pudong New Area (Shanghai), and the largest residential complex in the Pudong (Shanghai). Zoom-in for better details.
![](images/Beijing-Shanghai.jpg)

- **Nanjing-Hefei:** Visualization of UFZ inference results with geographically isolated areas (Nanjing and Hefei).
![](images/Nanjing-Hefei.jpg)

- **Visualization for comparison with different algorithms:** Comparison of details of functional areas identified by different methods in Beijing (Chaoyang District, Haidian District), Shanghai, Nanjing and Hefei.
![](images/VisualizationComparisonAlgorithm.jpg)

- **Confusion matrix for comparison with different algorithms:** Confusion matrix of all comparison experiments. 
![](images/ConfusionMatrixComparisonAlgorithm.jpg)

- VHR images, physical objects, semantic objects, and pie charts of these two objects in Beijing (Chaoyang District, Haidian District), Nanjing, and Hefei. 
![](images/DetailedCombinationFigure.jpg)


### 5. Sensitivity Analysis of G2MF <a name="discussion"></a>
- **The impact of deep learning models and graph structures on the proposed framework.**
![](images/FrameworkSensitivity.jpg)

### 6. Paper <a name="paper"></a>
**[A graph-based multi-modal data fusion framework for identifying urban functional zone]()**

Please cite the following paper if you find it useful for your research:
```
@article{tao2024g2mf,
  title={A graph-based multi-modal data fusion framework for identifying urban functional zone},
  author={Yuan Tao, Wanzeng Liu, Jun Chen, Jingxiang Gao, Ran Li, Jiaxin Ren, Shunxi Yin, Xiuli Zhu, Xinpeng Wang, Tingting Zhao, Xi Zhai, Ye Zhang, and Yunlu Peng},
  year={2024}
}
```

### 7. Outlook <a name="outlook"></a>
We are trying to produce national-scale UFZs, click **stars** to continue following.

### 8. License <a name="license"></a>
This repo is distributed under [GLPv3 License](https://www.gnu.org/licenses/gpl-3.0.en.html). The code can be used for academic purposes only.
