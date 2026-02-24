import os, shutil
import json


def create_folder_if_not_exists(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def empty_folder(folder):
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    except Exception as e:
      print('Failed to delete %s. Reason: %s' % (file_path, e))


def prepare_folder(folder):
  create_folder_if_not_exists(folder)
  empty_folder(folder)


def read_config(filepath):
  with open(filepath, 'r') as config_file:
    return json.load(config_file)


def store_to_json(data, filename):
  try:
    with open(filename, 'w', encoding='utf-8') as f:
      json.dump(data, f, indent=4, ensure_ascii=False)
  except Exception as e:
    print(f"An error occurred while storing data to {filename}: {e}")
