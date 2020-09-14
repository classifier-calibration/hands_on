# Prepares jupyter server and dependencies for  https://github.com/classifier-calibration/hands_on
FROM debian:testing-slim
MAINTAINER Classifier Calibration

# APPUSER parameters
ENV APPUSER appuser
ENV HOME_DIR /home/$APPUSER
ENV UID 1000

# Jupyter server parameters
ENV J_PORT 8889
ENV J_DATA $HOME_DIR/data


# Non-root user for server
RUN groupadd -g $UID $APPUSER
RUN useradd -r -m -d $HOME_DIR -u $UID -g $APPUSER $APPUSER

# Install OS dependencies and requirements
RUN apt-get update
RUN apt-get -y install jupyter-notebook python3-pip vim git

# Adding repo files to build image
ADD . $J_DATA/
WORKDIR $J_DATA
RUN pip3 install -r binder/requirements.txt
RUN pip3 install -e lib/PyCalib
RUN pip3 install -e lib/dirichlet_python
RUN rm -r $J_DATA

# jupyter configuration
USER $APPUSER
WORKDIR $HOME_DIR
RUN mkdir -p .jupyter
RUN echo '{\n\
	"NotebookApp": { \n\
		"password": "argon2:$argon2id$v=19$m=10240,t=10,p=8$0kpdY77Q82yVFOkbB8izXQ$F62qcyKIek9ehbm1c4KmMA" \n\
	}\n\
}' > $HOME_DIR/.jupyter/jupyter_notebook_config.json


# image fixed port
EXPOSE $J_PORT
USER $APPUSER
ENTRYPOINT /usr/bin/jupyter-notebook --port $J_PORT --no-browser --ip 0.0.0.0 --notebook-dir $J_DATA --allow-root



