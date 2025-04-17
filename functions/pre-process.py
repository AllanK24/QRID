from onnxruntime.quantization.preprocess import quant_pre_process

def preprocess_model(input_model_path, output_model_path, preprocess_args):
    try:
        print(f"Preprocessing model {input_model_path}...")
        quant_pre_process(input_model_path, output_model_path, **preprocess_args)
        print(f"Model {input_model_path} preprocessed successfully.")
    except Exception as e:
        print(f"Error preprocessing model {input_model_path}: {e}. Please check the input model path and preprocess arguments.")
        return
        