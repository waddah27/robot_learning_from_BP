# Other imports and helper functions
import os
import time
import mujoco
import itertools
import numpy as np
from PIL import Image
from mujoco import viewer
import json
# Graphics and plotting.
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import distutils.util
import subprocess

# load configs 
root_dir = os.path.dirname(__file__)
CONF = None
with open(os.path.join(root_dir,'configs.json')) as f:
  CONF = json.load(f)

def config_gpu():
  # Configure MuJoCo to use the EGL rendering backend (requires GPU)
  print('Setting environment variable to use GPU rendering:')
  load_dotenv()
  print(os.getenv('MUJOCO_GL'))


  # More legible printing from numpy.
  np.set_printoptions(precision=3, suppress=True, linewidth=100)

  from IPython.display import clear_output
  clear_output()


  # Set up GPU rendering.
  if subprocess.run('nvidia-smi').returncode:
    raise RuntimeError(
        'Cannot communicate with GPU. '
        'Make sure you are using a GPU Colab runtime. '
        'Go to the Runtime menu and select Choose runtime type.')

  # Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
  # This is usually installed as part of an Nvidia driver package, but the Colab
  # kernel doesn't install its driver via APT, and as a result the ICD is missing.
  # (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
  NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
  if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
      f.write("""{
      "file_format_version" : "1.0.0",
      "ICD" : {
          "library_path" : "libEGL_nvidia.so.0"
      }
  }
  """)

  # Check if installation was succesful.
  try:
    print('Checking that the installation succeeded:')
    mujoco.MjModel.from_xml_string('<mujoco/>')
  except Exception as e:
    raise e from RuntimeError(
        'Something went wrong during installation. Check the shell output above '
        'for more information.\n'
        'If using a hosted Colab runtime, make sure you enable GPU acceleration '
        'by going to the Runtime menu and selecting "Choose runtime type".')

  print('Installation successful.')

if __name__=='__main__':
  config_gpu()