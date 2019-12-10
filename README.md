# Chest X-ray Classification <br>Using Deep Learning
 ( Ref. [CheXNet Github](https://github.com/zoogzog/chexnet) / [CheXNet Paper](https://stanfordmlgroup.github.io/projects/chexnet/) )

----
### ToDo List

5. K-fold cross validation 적용<br/>
6. Categorical Training 적용<br/>
7. Other Network 적용<br/>
8. Transfer Learning 적용<br/>
10. Normal / Cardiomegaly / Lung / Pleural Categorical Training (for Localization) <br/>
~~1. TestSet 모두 Heatmap Image 생성~~ (Done)<br/>
~~2. Preprocessing 에서 Resize하고 Crop 확인~~ (Done) <br/>
~~3. Normalizer 확인~~ (Done)<br/>
~~4. 3가지 Normalize 방법 마다 차이 확인~~ (Done)<br/>
~~9. Post Processing에서 Activation Map의 Threshold 변경~~ (Done)<br/>

----

### 10-DEC-2019
* <b>Threshold 0.5 Visualize (05-DEC-2019 Model)</b><br>

<table>
  <thead>
  </thead>
<tbody>
    <tr>
        <th align="center">Normal</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Normal_0.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Normal_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Normal_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Normal_3.png" width="224px"/>
        </td>   
    </tr>
    <tr>
        <th align="center">Atelectasis</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Atelectasis_0.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Atelectasis_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Atelectasis_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Atelectasis_3.png" width="224px"/>
        </td>   
    </tr>
    <tr>
        <th align="center">Cardiomegaly</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Cardiomegaly_0.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Cardiomegaly_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Cardiomegaly_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Cardiomegaly_3.png" width="224px"/>
        </td>  
    </tr>
    <tr>
        <th align="center">Effusion</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Effusion_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Effusion_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Effusion_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Effusion_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Infiltration</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Infiltration_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Infiltration_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Infiltration_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Infiltration_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Mass</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Mass_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Mass_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Mass_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Mass_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Nodule</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Nodule_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Nodule_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Nodule_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Nodule_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Pneumonia</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pneumonia_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pneumonia_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pneumonia_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pneumonia_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Pneumothorax</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pneumothorax_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pneumothorax_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pneumothorax_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pneumothorax_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Consolidation</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Consolidation_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Consolidation_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Consolidation_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Consolidation_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Edema</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Edema_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Edema_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Edema_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Edema_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Emphysema</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Emphysema_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Emphysema_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Emphysema_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Emphysema_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Fibrosis</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Fibrosis_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Fibrosis_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Fibrosis_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Fibrosis_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Pleural_Thickening</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pleural_Thickening_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pleural_Thickening_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pleural_Thickening_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Pleural_Thickening_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Hernia</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Hernia_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Hernia_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Hernia_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/10-DEC-2019/heatmap_Hernia_3.png" width="224px"/>
        </td>
    </tr>
  </tbody>
</table>


* <b>Threshold 0.5 Visualize (Paper Model)</b><br>

<table>
  <thead>
  </thead>
<tbody>
    <tr>
        <th align="center">Normal</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Normal_0.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Normal_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Normal_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Normal_3.png" width="224px"/>
        </td>   
    </tr>
    <tr>
        <th align="center">Atelectasis</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Atelectasis_0.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Atelectasis_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Atelectasis_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Atelectasis_3.png" width="224px"/>
        </td>   
    </tr>
    <tr>
        <th align="center">Cardiomegaly</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Cardiomegaly_0.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Cardiomegaly_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Cardiomegaly_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Cardiomegaly_3.png" width="224px"/>
        </td>  
    </tr>
    <tr>
        <th align="center">Effusion</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Effusion_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Effusion_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Effusion_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Effusion_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Infiltration</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Infiltration_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Infiltration_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Infiltration_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Infiltration_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Mass</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Mass_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Mass_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Mass_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Mass_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Nodule</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Nodule_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Nodule_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Nodule_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Nodule_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Pneumonia</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Pneumonia_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Pneumonia_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Pneumonia_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Pneumonia_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Pneumothorax</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Pneumothorax_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Pneumothorax_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Pneumothorax_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Pneumothorax_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Consolidation</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Consolidation_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Consolidation_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Consolidation_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Consolidation_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Edema</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Edema_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Edema_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Edema_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Edema_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Emphysema</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Emphysema_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Emphysema_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Emphysema_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Emphysema_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Fibrosis</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Fibrosis_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Fibrosis_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Fibrosis_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Fibrosis_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Pleural_Thickening</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Pleural_Thickening_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Pleural_Thickening_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Pleural_Thickening_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Pleural_Thickening_3.png" width="224px"/>
        </td>
    </tr>
    <tr>
        <th align="center">Hernia</th>
        <th align="center"></th>
        <th align="center"></th>
        <th align="center"></th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/paper/heatmap_Hernia_0.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/paper/heatmap_Hernia_1.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Hernia_2.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/paper/heatmap_Hernia_3.png" width="224px"/>
        </td>
    </tr>
  </tbody>
</table>

----

### 05-DEC-2019

* 0 To 1 :: Train Loss = 0.154 | Test Batch Size = 16
* -1 To 1 :: Train Loss = 0.149 | Test Batch Size = 16

<table>
  <thead>
    <tr>
    <th align="center">Normal</th>
    <th align="center">Atelectasis</th>
    <th align="center">Cardiomegaly</th>
    <th align="center">Effusion</th>
    </tr>
  </thead>
<tbody>
    <tr>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_normal.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Atelectasis.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Cardiomegaly.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Effusion.png" width="224px"/>
        </td>   
    </tr>
    <tr>
        <th align="center">Infiltration</th>
        <th align="center">Mass</th>
        <th align="center">Nodule</th>
        <th align="center">Pneumonia</th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Infiltration.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Mass.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Nodule.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Pneumonia.png" width="224px"/>
        </td>   
    </tr>
    <tr>
        <th align="center">Pneumothorax</th>
        <th align="center">Consolidation</th>
        <th align="center">Edema</th>
        <th align="center">Emphysema</th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Pneumothorax.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Consolidation.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Edema.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Emphysema.png" width="224px"/>
        </td>  
    </tr>
    <tr>
        <th align="center">Fibrosis</th>
        <th align="center">Pleural_Thickening</th>
        <th align="center">Hernia</th>
    </tr>
    <tr>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Fibrosis.png" width="224px"/>
        </td> 
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Pleural_Thickening.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/05-DEC-2019/heatmap_Hernia.png" width="224px"/>
        </td>
    </tr>
  </tbody>
</table>


| Pathology     |AUROC <br>(CIFAR Norm)   | AUROC <br>(0 to 1 Norm)| AUROC <br>(-1 To 1 Norm)
| ------------- |:-------------:|:--------------:|:--------------:|
| Atelectasis   | 0.825         | 0.804          | 0.823          |
| Cardiomegaly  | 0.896         | 0.910          | 0.903          |
| Effusion      | 0.883         | 0.873          | 0.883          |
| Infiltration  | 0.707         | 0.702          | 0.709          |
| Mass          | 0.855         | 0.838          | 0.852          |
| Nodule        | 0.783         | 0.763          | 0.790          |
| Pneumonia     | 0.764         | 0.748          | 0.770          |
| Pneumothorax  | 0.872         | 0.855          | 0.878          |
| Consolidation | 0.812         | 0.800          | 0.816          |
| Edema         | 0.900         | 0.889          | 0.900          |
| Emphysema     | 0.932         | 0.899          | 0.931          |
| Fibrosis      | 0.851         | 0.809          | 0.840          |
| P.T.          | 0.783         | 0.767          | 0.789          |
| Hernia        | 0.930         | 0.942          | 0.938          |
| <b>Total AUROC  | <b>0.842    | <b>0.828       | <b>0.844       |
<br>

* Next Training ... Done
* Normalize -1 ~ 1 적용.
* Training :: Batch 256, Epoch 100
----
### 04-DEC-2019
* Image별 Normalize 적용 (0~1) -> Next Normalize (-1~1)
* Training :: Batch 192, Epoch 100, 33h 소요.
* 기존 Constant Value에서 각각의 연산이 추가되어 트레이닝 시간이 길어짐.
* Activation Map with Threshold
* Threshold 0.5 & 0.8 ( 50% & 80% )
<table>
<thead>
  <tr>
  <th align="center">Origin Image</th>
  <th align="center">Threshold 0.5</th>
  <th align="center">Threshold 0.8</th>
  </tr>
</thead>
<tbody>
    <tr>
        <td align="center">
        <img src="web/img/00009285_000.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/heatmap_threshold_0.5.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/heatmap_threshold_0.8.png" width="224px"/>
        </td>  
    </tr>
  </tbody>
</table>

----

### 28-NOV-2019
 * Training :: Batch 128, Epoch 150으로 16h 소요.
 * ImageNet에서의 Pretraining 된 Network를 가져와서 적용.

<table>
<thead>
  <tr>
  <th align="center">Origin Image</th>
  <th align="center">Image of the model I trained</th>
  <th align="center">Image with CheXNet Model</th>
  </tr>
</thead>
<tbody>
    <tr>
        <td align="center">
        <img src="web/img/00009285_000.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/heatmap_mine.png" width="224px"/>
        </td>
        <td align="center">
        <img src="web/img/heatmap.png" width="224px"/>
        </td>  
    </tr>
  </tbody>
</table>

| Pathology     |AUROC (Mine)   | AUROC (CheXNet)|
| ------------- |:-------------:|:--------------:|
| Atelectasis   | 0.825         | 0.832          |
| Cardiomegaly  | 0.896         | 0.910          |
| Effusion      | 0.883         | 0.886          |
| Infiltration  | 0.707         | 0.714          |
| Mass          | 0.855         | 0.865          |
| Nodule        | 0.783         | 0.803          |
| Pneumonia     | 0.764         | 0.765          |
| Pneumothorax  | 0.872         | 0.885          |
| Consolidation | 0.812         | 0.815          |
| Edema         | 0.900         | 0.901          |
| Emphysema     | 0.932         | 0.942          |
| Fibrosis      | 0.851         | 0.852          |
| P.T.          | 0.783         | 0.794          |
| Hernia        | 0.930         | 0.941          |
| <b>Total AUROC  | <b>0.842    | <b>0.850       | 
----