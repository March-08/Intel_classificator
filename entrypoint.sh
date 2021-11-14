#!/bin/bash

# general setting
BASE_DIR="/usr/local/class"

export PYTHONPATH=${PYTHONPATH}:/usr/lib/python3.9/site-packages:${BASE_DIR}/system/src
export LC_ALL=en_US.UTF8
export FLASK_APP=run.py

cd ${BASE_DIR}/src
gunicorn -b 0.0.0.0:8080 --workers 5 --limit-request-line 0 -t 1800 app:app 

