ARG OS=22.04
ARG CUDA_VERSION=12.2.2
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS} AS cuda

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \ 
            locate \
            lsb-release \
            sudo \
            libsm6 \
            libxext6 \
            libgl1-mesa-dev \
            libxcb-xkb-dev \
            libxcb-xkb1 \
            libxkbcommon-x11-0 \
            libdbus-1-dev \
            python3-pip \
            vim \
            curl \
            pybind11-dev \
            imagemagick \
        && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -U pip && pip install --no-cache-dir \
            torch \
            torchvision \
            pandas \
            numpy \
            tqdm \
            opencv-python \
            matplotlib \
            seaborn \
            torchmetrics \
            scikit-learn


ARG UNAME=tristarAI
ARG UID=1000
ARG GID=1000

ARG PKG_DIR=TristarAI_Takehome
ARG HOME_DIR=/home/$UNAME
ARG WS_DIR=${HOME_DIR}/takehome_ws


RUN groupadd -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME
RUN chown $UID:$GID /home/$UNAME

RUN echo "$UNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $UNAME

# Copy bashrc additions to enhance terminal usage
RUN cp /etc/skel/.bash_logout /etc/skel/.bashrc /etc/skel/.profile /home/$UNAME/
COPY --chown=$UNAME ${PKG_DIR}/docker/.git-prompt.sh ${PKG_DIR}/docker/.bash_prompt /home/$UNAME/
COPY --chown=$UNAME ${PKG_DIR}/docker/bashrc_additions /home/$UNAME/bashrc_additions
RUN cat /home/$UNAME/bashrc_additions >> /home/$UNAME/.bashrc && rm /home/$UNAME/bashrc_additions

COPY --chown=$UNAME ${PKG_DIR}/docker/entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
