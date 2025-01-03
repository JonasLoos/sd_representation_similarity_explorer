from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from sdhelper import SD
import numpy as np
from pathlib import Path
import PIL.Image
import hashlib
from flask import current_app
from PIL import Image
import io
from threading import Lock
import json
import traceback
import sys



app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = Path('images')
CACHE_FOLDER = Path('representations')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CACHE_FOLDER'] = CACHE_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure upload and cache folders exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
CACHE_FOLDER.mkdir(exist_ok=True)

# Global variable to store the loaded SD model
loaded_sd_model = None
sd_model_lock = Lock()


@app.route('/generate_representations', methods=['POST'])
def generate_representations():
    global loaded_sd_model

    # get parameters
    filename = request.form.get('filename')
    model_short = request.form.get('model')
    step = request.form.get('step')

    # check parameters
    if filename is None or filename == '':
        return jsonify({'status': 'error', 'msg': 'No filename provided'}), 400
    filepath = UPLOAD_FOLDER / f'{filename.strip()}.png'
    if model_short not in SD.known_models:
        return jsonify({'status': 'error', 'msg': 'Invalid model selected'}), 400
    if not filepath.is_file():
        return jsonify({'status': 'error', 'msg': 'Invalid file path or file type'}), 400
    if step is None or not step.isdigit():
        return jsonify({'status': 'error', 'msg': 'Invalid step provided'}), 400

    # generate representations if not cached
    cache_path = CACHE_FOLDER / filepath.stem / model_short / step
    if not cache_path.exists():
        with sd_model_lock:

            # load model
            if loaded_sd_model is None or loaded_sd_model.model_name != model_short:
                try:
                    loaded_sd_model = SD(model_short, disable_progress_bar=True, local_files_only=True)
                    if 'FLUX' in loaded_sd_model.model_name:
                        loaded_sd_model.quantize(['transformer', 'text_encoder_2'], model_cpu_offload=True)
                except Exception as e:
                    return jsonify({'status': 'error', 'msg': f'Failed to load model: {str(e)}'}), 500

            # generate representations
            model_info = json.load(open('static/model_info.json'))
            n = int(next(x for x in model_info if x['short'] == model_short)['default_image_shape'][1])
            image = PIL.Image.open(filepath).resize((n,n))
            representations = loaded_sd_model.img2repr(image, extract_positions=loaded_sd_model.available_extract_positions, step=int(step), seed=42)

            # Cache the representations
            cache_path.mkdir(parents=True, exist_ok=True)
            for pos, repr in representations.data.items():
                with open(cache_path / f'{pos}.bin', 'wb') as f:
                    f.write(repr.squeeze(0).permute(1, 2, 0).numpy().astype(np.float16).tobytes())

    return jsonify({'status': 'success'})


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'msg': 'No file part'}), 400

    file = request.files['file']
    if file.filename is None or file.filename == '':
        return jsonify({'status': 'error', 'msg': 'No selected file'}), 400

    # convert image to PNG format and center crop to square it
    image = Image.open(file.stream).convert('RGB')
    size = min(image.size)
    image = image.crop((
        (image.width - size) // 2,
        (image.height - size) // 2,
        (image.width + size) // 2,
        (image.height + size) // 2
    ))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # generate a hash-based filename and save the image
    file_hash = hashlib.md5(img_byte_arr).hexdigest()
    filename = f"{file_hash}.png"
    file_path = Path(app.config['UPLOAD_FOLDER']) / filename
    file_path.write_bytes(img_byte_arr)
    
    return jsonify({'status': 'success', 'filename': file_hash}), 200


@app.route('/cached_images.json', methods=['GET'])
def get_cached_images():
    cached_files = {
        img_file.stem: {
            model_path.name: {
                step_folder.name: [rep.stem for rep in step_folder.glob('*.bin')]
                for step_folder in model_path.glob('*') if step_folder.is_dir()
            }
            for model_path in (CACHE_FOLDER / img_file.stem).glob('*') if model_path.is_dir()
        }
        for img_file in UPLOAD_FOLDER.glob('*')
    }

    return jsonify({
        'cached_images': cached_files,
        'total_cache_size': sum(file.stat().st_size for file in CACHE_FOLDER.glob('**/*'))
    })


@app.route('/representations/<path:img_name>/<path:model_short>/<int:step>/<path:position>.bin', methods=['GET'])
def get_representation(img_name, model_short, step, position):
    cache_path = Path(app.config['CACHE_FOLDER'])
    file_path = cache_path / img_name / model_short / str(step) / f'{position}.bin'
    # current_app.logger.info(f'Downloading representation: {file_path}')

    if not file_path.exists() or not file_path.is_file():
        return jsonify({'error': 'File not found'}), 404

    if file_path.suffix != '.bin':
        return jsonify({'error': 'Invalid file type'}), 400

    return send_file(file_path, as_attachment=True)


@app.route('/images/<path:img_name>.png', methods=['GET'])
def get_image(img_name):
    file_path = Path(app.config['UPLOAD_FOLDER']) / f'{img_name}.png'
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(app.config['UPLOAD_FOLDER']), f'{img_name}.png')
    else:
        return jsonify({'error': 'Image not found'}), 404


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_index(path):
    if app.static_folder is None:
        return jsonify({'error': 'Static folder not found'}), 500

    # serve index.html by default
    if path == "":
        return send_from_directory(app.static_folder, 'index.html')

    # serve other files from the static folder
    file_path = Path(app.static_folder) / path
    if file_path.exists() and file_path.is_file():
        return send_from_directory(app.static_folder, path)
    else:
        return jsonify({'error': 'File not found'}), 404




if __name__ == '__main__':
    if "prod" in sys.argv:
        print("Starting server in production mode...")
        import waitress
        waitress.serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True)
