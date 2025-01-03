import json
import torch
from sdhelper import SD
import gc



def get_representation_info(sd):
    repr_shapes, image_shape = sd.get_representation_shapes()
    return {
        pos: {
            "channels": shape[-3],
            "spatial": shape[-1],
        }
        for pos, shape in repr_shapes.items()
    }, image_shape


def generate_model_info():
    model_info = []

    for short_name, config in SD.known_models.items():
        print(f"Processing {short_name}...")
        
        try:
            sd = SD(short_name, disable_progress_bar=True, local_files_only=True)
        except Exception as e:
            print(f"Error loading {short_name}: {e}")
            continue

        info = {
            "short": short_name,
            "name": config['name'],
            "guidance_scale": config.get('guidance_scale', 0.0),
        }

        try:
            info["representations"], info["default_image_shape"] = get_representation_info(sd)
        except Exception as e:
            print(f"Error getting representation info for {short_name}: {e}")
            info["representations"] = {}

        model_info.append(info)

        # Unload the model and free up memory
        del sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Processed {short_name} successfully.")

    return model_info



if __name__ == "__main__":
    print("Generating model info...")
    model_info = generate_model_info()
    
    with open("static/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print("model_info.json has been generated successfully.")
