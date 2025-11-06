# from flask import Flask
# import os
# from dotenv import load_dotenv
# from extensions import init_extensions
# from routes.index_routes import index_bp
# from routes.query_routes import query_bp
# from routes.api_routes import api_bp
# from routes.admin_routes import admin_bp

# load_dotenv()

# def create_app():
#     app = Flask(__name__)
#     app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
    
#     # Initialize extensions
#     init_extensions(app)
    
#     # Register blueprints
#     app.register_blueprint(index_bp)
#     app.register_blueprint(query_bp)
#     app.register_blueprint(api_bp)
#     app.register_blueprint(admin_bp)
    
#     return app

# if __name__ == '__main__':
#     app = create_app()
#     app.run(host='0.0.0.0', port=9000, debug=True)



from flask import Flask
import os
from dotenv import load_dotenv
from extensions import init_extensions
from routes.index_routes import index_bp
from routes.query_routes import query_bp
from routes.api_routes import api_bp
from routes.admin_routes import admin_bp
from error_handlers import register_error_handlers

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
    
    # Initialize extensions
    init_extensions(app)
    
    # Register blueprints
    app.register_blueprint(index_bp)
    app.register_blueprint(query_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(admin_bp)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=9000, debug=True)