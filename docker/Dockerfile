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
RUN echo "this step will be executed again if the value of FooBar Bar changes"
RUN pip install pillow==6.2.1 mmcv==0.4.3 pycocotools lxml
COPY download_model.sh . 
RUN ./download_model.sh
WORKDIR / 
RUN pip install gdown && gdown "https://drive.google.com/u/0/uc?id=1-QieHkR1Q7CXuBu4fp3rYrvDG9j26eFT"
RUN git clone https://github.com/catwhiskers/CascadeTabNet.git 
COPY train /
ENV PATH="/:${PATH}"

