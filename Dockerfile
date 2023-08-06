FROM nvcr.io/nvidia/pytorch:23.03-py3

# Setup working directory
RUN mkdir /TTS_Ground

WORKDIR /TTS_Ground

COPY requirements.txt /TTS_Ground/requirements.txt

# Install packages - Use cache dependencies 
RUN pip install --no-cache-dir -r requirements.txt

# Run our project exposed on port 8080
CMD ["python", "trainer.py"]