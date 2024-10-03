import os
import uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 允许所有域名进行访问

# 模型路径字典
models = {
    '追踪': 'best.pt',
    '检测': 'best.pt',
    '分类': 'models/classify/yolov8n-cls.pt',
    '姿势': 'models/pose/yolov8n-pose.pt',
    '分割': 'models/segment/yolov8n-seg.pt'
}

# 模型实例缓存
model_instances = {}

def load_model(model_path):
    """加载指定路径的模型，如果加载失败返回None"""
    try:
        return YOLO(model_path)  # 加载模型
    except Exception as e:
        print(f"加载模型失败: {model_path}, 错误: {e}")
        return None

@app.after_request
def after_request(response):
    """设置响应头，允许跨域访问"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def home():
    return "欢迎访问 YOLOv8 Flask 应用！"

@app.route('/request', methods=['POST'])
def handle_request():
    try:
        # 获取请求中选择的模型
        selected_model = request.form.get('model')
        if selected_model not in models:
            return jsonify({'error': '选择的模型无效。'}), 400

        # 获取模型路径，并加载模型
        model_path = models[selected_model]
        if selected_model not in model_instances:
            model_instances[selected_model] = load_model(model_path)
        model = model_instances[selected_model]

        if model is None:
            return jsonify({'error': '模型加载失败。'}), 500

        # 获取上传的图片文件
        img = request.files.get('img')
        if img is None:
            return jsonify({'error': '未提供图片。'}), 400

        # 生成唯一的文件名并保存图片
        img_name = str(uuid.uuid4()) + '.jpg'
        img_path = os.path.join('img', img_name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path)

        # 预测并保存结果
        save_dir = 'runs/detect'
        os.makedirs(save_dir, exist_ok=True)
        results = model.predict(img_path, save=True, project=save_dir, name=img_name.split('.')[0], device='cpu')

        # 构建结果图片的路径
        predicted_img_path = os.path.join(save_dir, img_name.split('.')[0], img_name)
        predicted_img_path = predicted_img_path.replace('\\', '/')  # 替换反斜杠为正斜杠，确保路径格式正确

        print(f"Serving image from: {predicted_img_path}")

        if not os.path.exists(predicted_img_path):
            return jsonify({'error': '预测结果未找到。'}), 500

        # 返回结果图片的URL路径
        return jsonify({'message': '预测成功！', 'img_path': f'/get/{img_name.split(".")[0]}/{img_name}'})
    except Exception as e:
        return jsonify({'error': f'处理请求时发生错误: {e}'}), 500

@app.route('/get/<folder>/<filename>', methods=['GET'])
def get_file(folder, filename):
    try:
        # 构建结果图片的完整路径
        save_dir = 'runs/detect'
        predicted_img_path = os.path.join(save_dir, folder, filename)

        if not os.path.exists(predicted_img_path):
            return jsonify({'error': '预测结果未找到。'}), 404

        # 发送结果图片文件
        return send_file(predicted_img_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': f'获取结果图片失败。错误: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # 允许从任何网络接口访问
