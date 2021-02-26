FROM quay.io/fenicsproject/stable:current

# Requirements for codebase
COPY requirements.txt /tmp/requirements.txt
COPY requirements-upgrade.txt /tmp/requirements-upgrade.txt
RUN apt upgrade -y && \
    apt update -y && \
    apt install -y gmsh && \
    pip install --upgrade -r /tmp/requirements.txt && \
    pip install --upgrade -r /tmp/requirements-upgrade.txt && \
    rm /tmp/requirements.txt && \
    rm /tmp/requirements-upgrade.txt && \
    pip install --upgrade jupyter_contrib_nbextensions jupyter_nbextensions_configurator && \
    jupyter notebook --generate-config && \
    jupyter contrib nbextension install --system && \
    jupyter nbextensions_configurator enable --system && \
    echo "c.NotebookApp.ip = '0.0.0.0'\n" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True\n" >> /root/.jupyter/jupyter_notebook_config.py


# Install CGAL and SVMTK
ARG CGAL_VERSION
ENV CGAL_VERSION 5.0.2
RUN apt update && apt install -y \
    bzip2 \
    cmake \
    curl \
    g++ \
    libboost-all-dev \
    libeigen3-dev \
    libgmp3-dev \
    libmpfr-dev \
    xz-utils \
    zlib1g-dev \
    git
RUN curl -sL https://github.com/CGAL/cgal/releases/download/releases/CGAL-${CGAL_VERSION}/CGAL-${CGAL_VERSION}.tar.xz | tar xpvfJ - &&\
    cd CGAL-${CGAL_VERSION} && \
    cmake -DWITH_Eigen3:BOOL=ON . && \
    make && \
    make install 
RUN pip install setuptools git+https://github.com/SVMTK/SVMTK.git


# Install Freesurfer dependencies
RUN apt-get update && \
    apt-get -y install tcsh tar wget libgomp1 perl-modules bc binutils perl psmisc unzip uuid-dev vim-common libjpeg62-dev