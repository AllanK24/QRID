from ultralytics import YOLO
from functions.constants import default_benchmark_kwargs
def benchmark(model:YOLO, kwargs:dict=default_benchmark_kwargs):
    """Perform benchmark on the given model. This function is used to evaluate the performance of the model on a specific dataset.

    Args:
        model (YOLO): Model to be benchmarked. This should be a valid YOLO model object.
        kwargs (dict, optional): A dictionary of arguments to be passed to the benchmark function. This allows for customization of the benchmarking process.
        Defaults to default_benchmark_kwargs.

    Returns:
        dict: A dictionary containing the results of the benchmark. This includes various performance metrics such as mAP, FPS, and other relevant statistics.
    """
    
    if not isinstance(model, YOLO):
        print("Model is not a YOLO object. Please provide a valid YOLO model.")
        return
        
    # Merge default benchmark arguments with user-provided arguments
    kwargs = {**default_benchmark_kwargs, **kwargs}
    
    # Perform benchmark
    print(f"Performing benchmarking...")
    benchmark_results = model.val(**kwargs)
    print(f"Benchmark completed.")
    print(f"Benchmark results: {benchmark_results}")
    
    return benchmark_results