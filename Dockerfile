# I recommend setting the CUDA image version to the same one supported by your GPU(s).
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
ENV PYTHONUNBUFFERED True
ENV DEBIAN_FRONTEND noninteractive
ENV TZ Etc/GMT

WORKDIR /vid-gptq
COPY . .

RUN apt-get update && apt-get upgrade -y

# Remove the below line if you are using Cloud Run, as you have set the base image to a Python image.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip

RUN apt-get update && apt-get install -y gcc build-essential cmake
RUN pip install -r requirements.txt

# Use cu117 if you are on CUDA 11.7.
RUN pip install auto-gptq==0.4.2 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
RUN pip install transformers==4.33.1 optimum==1.13.1

# Uncomment the below line to install the nano text editor for debugging (only useful in GCE Virtual Machines).
# RUN apt-get update && apt-get install -y nano

WORKDIR /vid-gptq/src

# For Firebase Admin SDK authentication users using GCE:
#   Change the below line to the absolute path to your Firebase Admin SDK Service Account key file.
# For others:
#   Remove the below line. (If you are using Cloud Run, this is already set up for you)
ENV GOOGLE_APPLICATION_CREDENTIALS "/vid-gptq/NOCOMMIT/Keys/admin-sdk-key.json"

CMD exec uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1