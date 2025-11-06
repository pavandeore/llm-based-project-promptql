from flask import render_template
from utils.helpers import handle_error

def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(error):
        return render_template('error.html', error="Page not found"), 404

    @app.errorhandler(500)
    def internal_error(error):
        return handle_error(error, "Internal server error")