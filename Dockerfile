# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch

# Set the working directory
WORKDIR /dnncanyon

# Copy the current directory contents into the container
COPY ./  /dnncanyon

# Install any needed packages specified in requirements.txt
RUN pip install -r ./requirements.txt

# Run when the container launches
CMD ["python", "inference.py"]