### 运行方法：在根目录创建python脚本，复制以下代码即可运行
```angular2html
from model import URetinex_Net
import cv2
import os

img = cv2.imread('demo/inputs/img1.png')
im_shape = (img.shape[1]//16*16, img.shape[0]//16*16)
img = cv2.resize(img, im_shape)
model = URetinex_Net()
out = model(img)
cv2.imwrite('tmp.jpg',out)
```

### 结果对比
 <div class="half" style="text-align: center;">
   <img src="../../demo/inputs/bicycle.jpg" width="400"/> <img src="../../demo/outputs/URetinex_Net/bicycle.jpg" width="400"/>
</div>
 <div class="half" style="text-align: center;">
   <img src="../../demo/inputs/cat.jpg" width="400"/> <img src="../../demo/outputs/URetinex_Net/cat.jpg" width="400"/>
</div>
 <div class="half" style="text-align: center;">
   <img src="../../demo/inputs/dog.jpg" width="400"/> <img src="../../demo/outputs/URetinex_Net/dog.jpg" width="400"/>
</div>






