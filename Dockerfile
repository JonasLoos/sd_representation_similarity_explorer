# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files from the representation_similarity_explorer
COPY ./representation_similarity_explorer/requirements.txt /app/
COPY ./representation_similarity_explorer/app.py /app/
COPY ./representation_similarity_explorer/generate_model_info.py /app/
COPY ./representation_similarity_explorer/static /app/static

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install sdhelper
COPY ./sdhelper /app/sdhelper
RUN pip install -e /app/sdhelper

# Install models (optional, modify this list as needed)
ARG MODELS="SD1.5"
ENV MODELS=$MODELS
RUN python -c "from sdhelper import SD; from huggingface_hub import snapshot_download; import os; models = os.environ.get('MODELS').split(','); [snapshot_download(SD.known_models[SD(model.strip(), config={'load_fn': lambda **_: None}).model_name]['name']) for model in models if model.strip()]"

# Generate model info
RUN python generate_model_info.py

# assume that the rust webworker is already built, otherwise run the following:
# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"
# WORKDIR /app/static/worker
# RUN wasm-pack build --target web --release --no-typescript --no-pack
# WORKDIR /app

# cleanup
RUN rm -rf /root/.cache/pip

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py", "prod"]
