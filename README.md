# Representation similarity explorer

An interactive tool for visualizing representation similarities in Stable Diffusion models. Supports multiple SD models, similarity measures, and custom image uploads (local version).

🔗 Live Demo: [sd-similarities.jloos.de](https://sd-similarities.jloos.de)  
🚀 Run Locally: See setup instructions below.

To use the tool, select one or more images to analyze. Now, hover over the images or similarity maps to show the similarities to the token at the current cursor position. For a good first example of interesting semantic correspondences, select two images that contain a human or animal and hover over the position of an eye. For further exploration change the SD model, the block (U-Net position), the similarity measure, and the noise level.


## Setup

```bash
# install requirements
pip install -r requirements.txt

# generate model info
python generate_model_info.py
```

You need to have the diffusion models from Hugging Face installed locally, that you want to use in the app. The simplest way to do that is by running this python code (add or remove models as you like):

```python
from sdhelper import SD
SD('SD1.5')
SD('SD3')
...
```

For a list of available models, check the sdhelper repository: https://github.com/JonasLoos/sdhelper

To rebuild the Rust webworker, install rust and read the webworker [readme](./static/worker/README.md).


## Usage

```bash
python app.py

# alternatively, for a production build:
python app.py prod
```

then go to http://localhost:5000


## Docker

Build the docker image (from the master thesis root folder):

```bash
docker build -t representation-similarity-explorer -f Dockerfile .
```

Run the docker container:

```bash
docker run -p 5000:5000 representation-similarity-explorer
```


## Additional Information

This project is based on the [h-space-similarity-explorer](https://github.com/JonasLoos/h-space-similarity-explorer) and was developed as part of [my master thesis](https://github.com/JonasLoos/thesis) at TU Berlin.

Architecturally, the representation similarity explorer is split into a Flask-based backend that computes the representations for uploaded images using the `sdhelper` package (https://github.com/JonasLoos/sdhelper) and a simple frontend built with HTML, CSS, and JavaScript. To improve the performance and interactivity, an asynchronous webworker in Rust, that computes the shown similarities in the browser in real time, is used.


## TODO

* [ ] add support for different aspect ratios and image sizes
* [ ] fix representation generation for SDXL Lightning models
* [ ] add image generation
* [ ] add nice error msg when model is not downloaded
* [ ] support for model download from the web interface
