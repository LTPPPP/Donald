from flask import Blueprint, render_template

templates_bp = Blueprint('templates_bp', __name__)

@templates_bp.route('/')
def index():
    return render_template('index.html')

@templates_bp.route('/course')
def course():
    return render_template('course.html')

@templates_bp.route('/login_page')
def login_page():
    return render_template('login.html')

@templates_bp.route('/course-child')
def course_child():
    return render_template('course-child.html')

@templates_bp.route('/course-parent')
def course_parent():
    return render_template('course-parent.html')

@templates_bp.route('/survey')
def survey():
    return render_template('survey.html')
