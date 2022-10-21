### 运行方法：在根目录创建python脚本，复制以下代码即可运行
```angular2html
from model import SCI
import cv2
import os

img = cv2.imread('demo/inputs/img1.png')
im_shape = (img.shape[1]//16*16, img.shape[0]//16*16)
img = cv2.resize(img, im_shape)
model = SCI()
out = model.run(img)
cv2.imwrite('tmp.jpg',out)
```

### 结果对比
 <div class="half" style="text-align: center;">
   <img src="../../demo/inputs/img1.png" width="400"/> <img src="../../demo/outputs/SCI/img1.png" width="400"/>
</div>
 <div class="half" style="text-align: center;">
   <img src="../../demo/inputs/img2.png" width="400"/> <img src="../../demo/outputs/SCI/img2.png" width="400"/>
</div>
 <div class="half" style="text-align: center;">
   <img src="../../demo/inputs/img4.png" width="400"/> <img src="../../demo/outputs/SCI/img4.png" width="400"/>
</div>






