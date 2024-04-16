# # Ubuntu 20.04 base image
FROM ubuntu:20.04

# # Manually download and install libgl1-mesa-glx
RUN apt-get update
# RUN pip install opencv-python
RUN apt install -y libgl1-mesa-glx
# Grant write permissions to /tmp directory
RUN chmod 1777 /tmp

#python 3.11 lambda base image
FROM public.ecr.aws/lambda/python:3.8

RUN pip3 install opencv-python-headless

#copy requirements.txt to container
COPY requirements.txt ./

#installing dependencies
RUN pip3 install -r requirements.txt

# Install libgl1-mesa-glx using apk package manager
# RUN apk add --no-cache mesa-dri-swrast

#copy function code to container
COPY src/lambda_function.py ./

#setting the CMD to your handler file_name.function_name
CMD ["lambda_function.lambda_handler"]