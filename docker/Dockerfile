ARG BASE_IMG=${BASE_IMG}
FROM ${BASE_IMG} 
RUN pip install -q mmcv terminaltables
RUN git clone --branch v1.2.0 'https://github.com/open-mmlab/mmdetection.git'
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/optional.txt
RUN python setup.py install 
RUN python setup.py develop
RUN pip install -r requirements.txt 
RUN pip install pillow==6.2.1 mmcv==0.4.3 pycocotools
COPY download_model.sh . 
RUN ./download_model.sh 


