
# In[ ]:

from IPython import get_ipython
# Installations and Imports
get_ipython().system('pip install pillow')
get_ipython().system('pip install selenium')
# https://github.com/AndrewCarterUK/pascal-voc-writer
get_ipython().system('pip install pascal-voc-writer')
# https://github.com/gereleth/jupyter-bbox-widget
get_ipython().system('pip install jupyter_bbox_widget')
# https://github.com/elisemercury/Duplicate-Image-Finder
get_ipython().system('pip install difPy')
get_ipython().system('apt-get update')
get_ipython().system('apt install chromium-chromedriver')
get_ipython().system('cp /usr/lib/chromium-browser/chromedriver /usr/bin')

import sys
import PIL
import os
import time
import requests
import io
import hashlib
from jupyter_bbox_widget import BBoxWidget
import ipywidgets as widgets
import json
import base64
from IPython.display import Image 
from pascal_voc_writer import Writer
from google.colab import files, output, drive
from selenium import webdriver
from selenium.webdriver.common.by import By
from difPy import dif

output.enable_custom_widget_manager()
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
wd = webdriver.Chrome('chromedriver',chrome_options=chrome_options)


# In[ ]:


def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    
    

    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"
  
    # load the page
    wd.get(search_url.format(q=query))
    print("page loaded")
    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements(By.CSS_SELECTOR,"img.Q4LuWd")
        # thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements(By.CSS_SELECTOR, 'img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            return
            load_more_button = wd.find_elements(By.CSS_SELECTOR, ".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

def persist_image(folder_path:str, url:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = PIL.Image.open(image_file).convert('RGB')
        if os.path.exists(folder_path):
            file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        else:
            os.mkdir(folder_path)
            file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=100)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

def search_and_download(search_term:str,number_images=5, target_path='./images'):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.1)
        
    for elem in res:
        persist_image(target_path, elem)

get_ipython().system('mkdir "/content/annotations"')

def encode_image(filepath, image_path='/content/images/'):
    filepath = image_path + filepath
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded

# when Skip button is pressed we move on to the next file
def on_skip(image_path='/content/images'):
  try:
    w_progress.value += 1
    # open new image in the widget
    image_file = files[w_progress.value]
    w_bbox.image = encode_image(image_file)

    w_bbox.bboxes = [] 
  except IndexError:
    print("No more images to annotate!")

  # delete skipped image
  get_ipython().system('rm {image_path}/{files[w_progress.value-1]}')

def on_submit(image_path='/content/images/', Save_PASCAL_VOC=True):
    image_file = files[w_progress.value]
    # save annotations for current image
    annotations[image_file] = w_bbox.bboxes
    
    if Save_PASCAL_VOC == True:
      save_pascal_voc_format(image_path, image_file)

    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=4)
    try:
      w_progress.value += 1
      # open new image in the widget
      image_file = files[w_progress.value]
      w_bbox.image = encode_image(image_file)
    
      w_bbox.bboxes = [] 
    except IndexError:
      print("No more images to annotate!")

def save_pascal_voc_format(image_path, image_file):
    image = PIL.Image.open(image_path + image_file)
    width, height = image.size
      
    writer = Writer(image_path + image_file, width, height)

    for annotation in annotations[image_file]:
      writer.addObject(annotation['label'], annotation['x'], annotation['y'], annotation['x'] + annotation['width'], annotation['y'] + annotation['height'])

    image_name = image_file.split('.')[0]
    writer.save(f'/content/annotations/{image_name}.xml')


# In[ ]:


search_term = input("Enter google search: ")
# The number of images to download
number_images = 20
search_and_download(search_term, number_images)


# In[ ]:


# Remove Duplicate Images
dif.compare_images("/content/images/", delete=True)


# In[ ]:


# Open Annotation Tool
files = os.listdir('/content/images/')

annotations = {}
annotations_path = 'annotations.json'
# a progress bar to show how far we got
w_progress = widgets.IntProgress(value=0, max=len(files), description='Progress')

# the bbox widget
w_bbox = BBoxWidget(
    image = encode_image(files[0]),
    classes=['plastic', 'metal', 'paper', 'glass'] # Add, Remove or change the classes according to your problem
)

w_container = widgets.VBox([
    w_progress,
    w_bbox,
])


w_bbox.on_skip(on_skip)
w_bbox.on_submit(on_submit)

w_container


# In[ ]:


drive.mount('/content/drive')

dataset_folder="MyDataset"
get_ipython().system('mkdir /content/drive/My\\ Drive/{dataset_folder}')
get_ipython().system('cp -r /content/images /content/drive/My\\ Drive/$dataset_folder')
get_ipython().system('cp -r /content/annotations /content/drive/My\\ Drive/$dataset_folder')
get_ipython().system('cat /content/annotations.json >> /content/drive/My\\ Drive/$dataset_folder/annotations.json')


# In[ ]:


get_ipython().system('rm -rf /content/images')
get_ipython().system('rm -rf /content/annotations')
get_ipython().system('rm /content/annotations.json')


