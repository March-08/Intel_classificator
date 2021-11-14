FROM tensorflow/tensorflow

COPY src/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

RUN mkdir -p /usr/local/class
COPY src /usr/local/class/src
COPY entrypoint.sh /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/bin/bash", "./entrypoint.sh"]
