# pupil-distance-detector

這是一個使用 OpenCV 與傳統影像處理方法實作的瞳孔距離偵測專案。  
目前 README 先以 `Pipeline V1` 為主，重點放在：

- 讀取 `data/input/` 內的人臉圖片
- 執行瞳孔偵測流程
- 將處理後的成果圖輸出到 `data/output/`
- 輸出左右瞳孔中心座標與像素距離

## 專案結構

```text
pupil-distance-detector/
├── data/
│   ├── input/
│   ├── output/
│   └── samples/
├── requirements.txt
├── pyproject.toml
├── README.md
├── main.py
└── src/
    └── pupil_distance_detector/
        ├── __init__.py
        ├── main.py
        ├── preprocessing/
        ├── edge_detection/
        ├── feature_detection/
        ├── measurement/
        ├── pipelines/
        │   ├── base.py
        │   ├── factory.py
        │   ├── pipeline_v1.py
        │   ├── pipeline_v2.py
        │   └── pipeline_v3.py
        └── utils/
```

## 建立虛擬環境

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 如果 `.venv` 已存在

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## 需要安裝的套件

[requirements.txt](c:/Users/GIGABYTE/Image%20processing/requirements.txt) 目前包含：

- `numpy`
- `opencv-python`
- `matplotlib`

安裝指令：

```powershell
pip install -r requirements.txt
```

## 圖片資料夾說明

- [data/input](c:/Users/GIGABYTE/Image%20processing/data/input)  
  放要拿來做瞳孔偵測的原始人臉圖片

- [data/output](c:/Users/GIGABYTE/Image%20processing/data/output)  
  放經過 pipeline 處理後輸出的成果圖片

- [data/samples](c:/Users/GIGABYTE/Image%20processing/data/samples)  
  放示範圖或文件用圖片

## Pipeline V1

[pipeline_v1.py](c:/Users/GIGABYTE/Image%20processing/src/pupil_distance_detector/pipelines/pipeline_v1.py) 目前採用的流程為：

1. 載入輸入影像並轉為灰階
2. 進行小角度旋轉搜尋
3. 使用 Haar cascade 偵測臉部與雙眼區域
4. 建立左右眼局部搜尋範圍
5. 對眼睛區域做 Gaussian Blur
6. 透過自適應二值化、Hough Circle、Contour 產生候選
7. 以暗度、圓形程度與局部黑色核心進行微調
8. 選出最佳左右瞳孔中心
9. 計算瞳孔中心距離
10. 產生標註後的輸出圖片

## 執行方式

### 使用專案根目錄入口

```powershell
python main.py --input data/input/1.jpg --output data/output/result_v1.jpg --pipeline v1
```

### 使用虛擬環境 Python

```powershell
.\.venv\Scripts\python.exe main.py --input data/input/1.jpg --output data/output/result_v1.jpg --pipeline v1
```

### 說明

- `--input`：輸入圖片路徑
- `--output`：輸出圖片路徑
- `--pipeline v1`：指定使用 `Pipeline V1`

## 輸出結果

執行完成後會有兩種結果：

1. 終端機輸出

```text
Pipeline: v1
Left pupil center: (x1, y1)
Right pupil center: (x2, y2)
Distance: xx.xx px
Saved annotated output to data/output/result_v1.jpg
```

2. 輸出圖片

輸入圖片經過 `Pipeline V1` 處理後，會儲存到你指定的輸出位置，例如：

```text
data/output/result_v1.jpg
```

也就是說，放在 [data/input](c:/Users/GIGABYTE/Image%20processing/data/input) 的圖片，經過 pipeline 後，成果圖會存到 [data/output](c:/Users/GIGABYTE/Image%20processing/data/output)。
