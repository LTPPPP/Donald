import sys
import os

# Add the directory containing your modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Blueprint, request, jsonify

auth_bp = Blueprint('auth_bp', __name__)

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Vui lòng nhập đầy đủ thông tin!'}), 400
    
    if email == "test@example.com" and password == "password":
        return jsonify({'message': 'Đăng Nhập thành công!'}), 200
    else:
        return jsonify({'error': 'Sai thông tin đăng nhập!'}), 401

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirm_password')
    
    if not email or not password or not confirm_password:
        return jsonify({'error': 'Vui lòng nhập đầy đủ thông tin!'}), 400
    
    if password != confirm_password:
        return jsonify({'error': 'Mật khẩu xác nhận không khớp!'}), 400
    
    return jsonify({'message': 'Đăng Ký thành công!'}), 200
