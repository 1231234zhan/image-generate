# Face-Attribute-Editing

使用CVAE-GAN实现人脸变形。使用dlib检测出人脸landmarks并作为prior condition。修改landmarks可以实现人脸变形。GAN用以增强生成图片细节

原图

![ori](img/ori.png)

使用原图生成原图

![syn](img/syn.png)

修改图片landmarks实现人脸变形（实际上更像是学习到部分特征的人脸替换）

![lsyn](img/lsyn.png)

