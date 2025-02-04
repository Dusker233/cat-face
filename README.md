# 中大猫谱猫脸识别

从猫谱小程序云存储拉取猫图，然后训练一个神经网络模型来识别猫猫。

## 前期准备

1. 每只猫猫积攒了超过 10 张高质量照片（照片清晰，且照片中只有一只猫猫）。
2. 虽然可以使用 CPU 进行神经网络的训练，但是速度较慢，因此最好准备一张具有足够显存的 GPU。（如果显存不够，你可以根据你的显存大小适当调整图像分辨率，但是可能会导致模型训练效果下降。）
3. 网络服务器一台，用于部署识别服务。（需要自备 web 后端技能。）服务器不需要 GPU。
4. 已注册备案域名一个，获取 SSL，用于小程序调用服务。

## 手把手教程

https://docs.qq.com/doc/DSHhmTkNNYUNYTkpH

## Quick Start

### 1. 训练模型

1. 在训练服务器（或你的个人 PC）上拉取本仓库代码。
2. 进入 `data` 目录，执行 `npm install` 安装依赖。（需要 Node.js 环境，不确定老版本 Node.js 兼容性，建议使用最新版本。）
3. 复制 `config.demo.ts` 文件并改名为 `config.ts`，填写Laf云环境的 `LAF_APPID`；在Laf云后台调试执行 `getTempCOS` 云函数，复制输出到 `config.ts` 中。
4. 执行 `npm start`，脚本将根据小程序数据库记录拉取小程序云存储中的图片。
5. 返回仓库根目录，执行 `python -m pip install -r requirements.txt` 安装依赖。（需要 Python>=3.8。不建议使用特别新版本的 Python，可能有兼容性问题。）
6. 执行 `bash prepare_yolov5.sh` 拉取YOLOv5目标检测模型所需的代码，然后下载并预处理模型数据。
7. 执行 `bash process.sh`，自动进行数据处理、数据集划分和模型训练的工作。工作完成后，你应当看到目录的 export 文件夹下存在如下四个文件：
    - `cat.onnx`；
    - `cat.json`；
    - `fallback.onnx`；
    - `fallback.json`。
<!-- 7. 执行`python3 data_preprocess.py`，脚本将使用YOLOv5从`data/photos`的图片中识别出猫猫并截取到`data/crop_photos`目录。
8. 执行`python3 main.py`，使用默认参数训练一个识别猫猫图片的模型。（你可以通过`python3 main.py --help`查看帮助来自定义一些训练参数。）程序运行结束时，你应当看到目录的export文件夹下存在`cat.onnx`和`cat.json`两个文件。（训练数据使用TensorBoard记录在`lightning_logs`文件夹下。若要查看准确率等信息，请自行运行TensorBoard。）
9. 执行`python3 main.py --data data/photos --size 224 --name fallback`，使用修改后的参数训练一个在YOLOv5无法找到猫猫时使用的全图识别模型。程序运行结束时，你应当看到目录的export文件夹下存在`fallback.onnx`和`fallback.json`两个文件。 -->

### 2. 部署服务
1. 在部署服务器上拉取本仓库代码。
2. 执行 `pip install -r requirements.txt` 安装依赖。（需要 Python>=3.8。不建议使用特别新版本的 Python，可能有兼容性问题。）
3. 执行 `bash prepare_yolov5.sh` 拉取YOLOv5目标检测模型所需的代码，然后下载并预处理模型数据。
4. 将 export 文件夹从训练服务器中拷贝到部署服务器中。export 文件夹中应包括四个文件：`cat.onnx`, `cat.json`, `fallback.onnx`, `fallback.json`。
5. 在仓库根目录中创建 `env` 文件（不需要 . 号开头），填写服务运行参数。示例：
    ```bash
    HOST_NAME=127.0.0.1 # 主机名
    PORT=3456 # HTTP服务端口

    SECRET_KEY='keyboard cat' # 接口密钥（用于哈希签名认证）
    TOLERANT_TIME_ERROR=60 # 调用接口时附带的时间戳参数与服务器时间之间的最大允许误差（单位：s）

    IMG_SIZE=256 # 猫猫识别模型的输入图像大小
    FALLBACK_IMG_SIZE=512 # 无法检测到猫猫时使用的全图识别模型的输入图像大小

    CAT_BOX_MAX_RET_NUM=5 # 接口返回的图片中检测到的猫猫的最大个数
    RECOGNIZE_MAX_RET_NUM=20 # 接口返回的猫猫识别结果候选列表的最大个数
    ```
6. 现在，执行 `python3 app.py`，HTTP 接口服务将被启动。🎉（你可以自行使用 gunicorn 等框架完成更规范的部署，也可以只使用 pm2 进行简单管理。）
