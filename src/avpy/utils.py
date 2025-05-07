import psutil
import time
import gc
import os
import sys
from datetime import datetime

# Memory profiling functions
def get_memory_usage():
    """Return the current memory usage of this process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

def log_memory_usage(label):
    """Log the current memory usage with a label."""
    mem_usage = get_memory_usage()
    timestamp = datetime.now().strftime("%H:%M:%S")
    #print(f"[{timestamp}] MEMORY ({label}): {mem_usage:.2f} MB")

def memory_report(func):
    """Decorator to report memory usage before and after function execution."""
    def wrapper(*args, **kwargs):
        gc.collect()  # Force garbage collection before measurement
        label_before = f"Before {func.__name__}"
        log_memory_usage(label_before)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        gc.collect()  # Force garbage collection before measurement
        label_after = f"After {func.__name__} (took {elapsed:.2f}s)"
        log_memory_usage(label_after)
        
        return result
    return wrapper

def log_variable_memory_usage(variables, label):
    """Log memory usage of specified variables."""
    #print(f"\n[{datetime.now().strftime('%H:%M:%S')}] MEMORY PROFILE ({label}):")
    for var_name, var_value in variables.items():
        size_mb = sys.getsizeof(var_value) / (1024 * 1024)  # Convert to MB
        print(f"  {var_name}: {size_mb:.2f} MB")
    #print()