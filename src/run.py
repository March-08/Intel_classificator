# -*- coding: utf-8 -*-

# Internal imports
from app import app

if __name__ == '__main__':
    app.run(debug=app.config.get('DEBUG', True))
