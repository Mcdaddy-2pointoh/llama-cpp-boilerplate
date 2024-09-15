# Imports 
import os
from huggingface_hub import hf_hub_list, hf_hub_download
from utils.misc import bcolors
import llama_cpp
from dotenv import dotenv_values

# Function to DOWNLOAD model
def get_model(repo_id: str, filename: str, save_dir: str, local_files_only: bool = False):
    """
    Function: Downloads only GGUF files from a model repository on Hugging Face Hub.
    Args:
        repo_id (str): The ID of the model on Hugging Face Hub (e.g., 'TheBloke/Llama-2-7B-GGUF').
        save_dir (str): The directory where the GGUF files will be stored.
    Returns: Saved model path and filename
    """

    # Clean the inputs
    repo_id = repo_id.strip()
    filename = filename.strip()
    save_dir = save_dir.strip()
    
    # Validate repo_id
    if len(repo_id.split('/')) == 2:
        raise ValueError(f"{bcolors.BOLD}{bcolors.FAIL}Invalid repo_id{bcolors.ENDC}: repo_id must be in the format `<organization>/<model_name>` found name as {bcolors.WARNING}{repo_id}{bcolors.ENDC}")
    
    # Else make the local repo dir
    else:
        local_repo_id = f"model--{'--'.join(repo_id.split('/'))}"

    # Create directory if it does not exist
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"{bcolors.BOLD}{bcolors.FAIL}Directory not found{bcolors.ENDC}: Could not locate {bcolors.WARNING}{save_dir}{bcolors.ENDC}")
    
    # If repo dir not found 
    elif not local_repo_id in os.listdir(save_dir):
        local_repo_path = os.path.join(save_dir, local_repo_id)
        os.mkdir(local_repo_path)

    # If non gguf filename is provided then raise error
    elif not filename.endswith('.gguf'):
        raise ValueError(f"{bcolors.BOLD}{bcolors.FAIL}Invalid file format{bcolors.ENDC}: Can only use models with `.gguf` extension, found .{bcolors.WARNING}{filename.split(".")[-1]}{bcolors.ENDC}")

    # Validate if file name is in the gguf dir
    else:    
        # Check if file is in the local repo
        if filename in os.listdir(local_repo_path):
            return {'status': True, 'model_path': os.path.join(local_repo_path, filename)}
        
        elif local_files_only:
            raise FileNotFoundError(f"{bcolors.BOLD}{bcolors.FAIL}GGUF file not found{bcolors.ENDC}: Could not find filename `{filename}` locally and flag `local_files_only` is set True, set flag {bcolors.WARNING}`local_files_only=False`{bcolors.ENDC}")

        # List all files in the model's repository 
        files = hf_hub_list(repo_id=repo_id)
    
        # check if filename in files 
        if not filename in files:
            raise FileNotFoundError(f"{bcolors.BOLD}{bcolors.FAIL}GGUF file not found{bcolors.ENDC}: Could not locate a gguf file format in repo {bcolors.WARNING}{repo_id}{bcolors.ENDC}")
        
        # Download the file if it exists        
        print(f"Found {filename} GGUF file. Downloading...")
        file_path = hf_hub_download(repo_id=repo_id, filename=filename.rfilename, local_dir=local_repo_path)
        return {'status': True, 'model_path': file_path}


# Load the model
def load_model(configs: dict, hub_path: str, local_files_only: bool = False):
    """
    Function: Downloads only GGUF files from a model repository on Hugging Face Hub.
    Args:
        repo_id (str): The ID of the model on Hugging Face Hub (e.g., 'TheBloke/Llama-2-7B-GGUF').
        save_dir (str): The directory where the GGUF files will be stored.
    Returns: Saved model path and filename
    """
    # Get model configs from config dictionary
    repo_id = configs['model']['repo_id']
    filename = configs['model']['filename']
    llama_cpp_configs = ['llama_cpp']['configurations']
    llama_cpp_params = llama_cpp_configs.keys()

    # get model path
    try:
        model_status, model_path = get_model(repo_id=repo_id, filename=filename, save_dir=hub_path, local_files_only=local_files_only)
        if not model_status:
            raise RuntimeError(f"{bcolors.BOLD}{bcolors.FAIL}Could not get model path{bcolors.ENDC}") 

    # Raise error if model path could not be found 
    except Exception as e:
        raise RuntimeError(f"{bcolors.BOLD}{bcolors.FAIL}Could not get model path{bcolors.ENDC}") from e

    # Instantiate the model using llama-cpp's Llama class
    try:
        model_obj = llama_cpp.Llama(model_path=model_path, 
                                n_gpu_layers = llama_cpp_configs['n_gpu_layers'] if 'n_gpu_layers' in llama_cpp_params else 0,
                                split_mode = llama_cpp_configs['split_mode'] if "split_mode" in llama_cpp_params else llama_cpp.LLAMA_SPLIT_MODE_LAYER,
                                main_gpu = llama_cpp_configs['main_gpu'] if 'main_gpu' in llama_cpp_params else 0,
                                tensor_split = llama_cpp_configs['tensor_split'] if 'tensor_split' in llama_cpp_params else None,
                                rpc_servers = llama_cpp_configs['rpc_servers'] if 'rpc_servers' in llama_cpp_params else None,
                                vocab_only = llama_cpp_configs['vocab_only'] if 'vocab_only' in llama_cpp_params else False,
                                use_mmap = llama_cpp_configs['use_mmap'] if 'use_mmap' in llama_cpp_params else True,
                                use_mlock = llama_cpp_configs['use_mlock'] if 'use_mlock' in llama_cpp_params else True,
                                kv_overrides = llama_cpp_configs['kv_overrides'] if 'kv_overrides' in llama_cpp_params else None,
                                seed = llama_cpp_configs['seed'] if 'seed' in llama_cpp_params else llama_cpp.LLAMA_DEFAULT_SEED,
                                n_ctx = llama_cpp_configs['n_ctx'] if 'n_ctx' in llama_cpp_params else 512,
                                n_batch = llama_cpp_configs['n_batch'] if 'n_batch' in llama_cpp_params else 512,
                                n_threads = llama_cpp_configs['n_threads'] if 'n_threads' in llama_cpp_params else None,
                                n_threads_batch = llama_cpp_configs['n_threads_batch'] if 'n_threads_batch' in llama_cpp_params else None,
                                rope_scaling_type = llama_cpp_configs['rope_scaling_type'] if 'rope_scaling_type' in llama_cpp_params else llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
                                pooling_type = llama_cpp_configs['pooling_type'] if 'pooling_type' in llama_cpp_params else llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,
                                rope_freq_base = llama_cpp_configs['rope_freq_base'] if 'rope_freq_base' in llama_cpp_params else 0,
                                rope_freq_scale = llama_cpp_configs['rope_freq_scale'] if 'rope_freq_scale' in llama_cpp_params else 0,
                                yarn_ext_factor = llama_cpp_configs['yarn_ext_factor'] if 'yarn_ext_factor' in llama_cpp_params else -1,
                                yarn_attn_factor = llama_cpp_configs['yarn_attn_factor'] if 'yarn_attn_factor' in llama_cpp_params else 1,
                                yarn_beta_fast = llama_cpp_configs['yarn_beta_fast'] if 'yarn_beta_fast' in llama_cpp_params else 32,
                                yarn_beta_slow = llama_cpp_configs['yarn_beta_slow'] if 'yarn_beta_slow' in llama_cpp_params else 1,
                                yarn_orig_ctx = llama_cpp_configs['yarn_orig_ctx'] if 'yarn_orig_ctx' in llama_cpp_params else 0,
                                logits_all = llama_cpp_configs['logits_all'] if 'logits_all' in llama_cpp_params else False,
                                embedding = llama_cpp_configs['embedding'] if 'embedding' in llama_cpp_params else False,
                                offload_kqv = llama_cpp_configs['offload_kqv'] if 'offload_kqv' in llama_cpp_params else True,
                                flash_attn = llama_cpp_configs['flash_attn'] if 'flash_attn' in llama_cpp_params else False,
                                last_n_tokens_size = llama_cpp_configs['last_n_tokens_size'] if 'last_n_tokens_size' in llama_cpp_params else 64,
                                lora_base = llama_cpp_configs['lora_base'] if 'lora_base' in llama_cpp_params else None,
                                lora_scale = llama_cpp_configs['lora_scale'] if 'lora_scale' in llama_cpp_params else 1,
                                lora_path =  llama_cpp_configs['lora_path'] if 'lora_path' in llama_cpp_params else None,
                                numa = llama_cpp_configs['numa'] if 'numa' in llama_cpp_params else False,
                                chat_format = llama_cpp_configs['chat_format'] if 'chat_format' in llama_cpp_params else None,
                                chat_handler = llama_cpp_configs['chat_handler'] if 'chat_handler' in llama_cpp_params else None,
                                draft_model = llama_cpp_configs['draft_model'] if 'draft_model' in llama_cpp_params else None,
                                type_k = llama_cpp_configs['type_k'] if 'type_k' in llama_cpp_params else None,
                                type_v = llama_cpp_configs['type_v'] if 'type_v' in llama_cpp_params else None,
                                spm_infill = llama_cpp_configs['spm_infill'] if 'spm_infill' in llama_cpp_params else False,
                                verbose = llama_cpp_configs['verbose'] if 'verbose' in llama_cpp_params else True
    )
        
        return model_obj
    
    except Exception as e:
        raise RuntimeError(f"{bcolors.BOLD}{bcolors.FAIL}Could not load the model, please validate the config file{bcolors.ENDC}") from e


