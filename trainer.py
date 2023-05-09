
# In[ ]:
from IPython import get_ipython

get_ipython().system('pip install imageai --upgrade')
get_ipython().system('pip install split-folders')

import PIL
import os
import splitfolders
from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection
from google.colab import drive


# In[ ]:


# Copy our dataset from google drive to google colab for faster training
drive.mount('/content/drive')
dataset_folder="MyDataset"
model_folder="MyModels"
directory_name = "MyWorkspace"

input_folder = f"/content/drive/MyDrive/{dataset_folder}"
output = f"/content/{directory_name}"
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.70, .15, .15))

# Download pretrained model
get_ipython().system('wget https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5 /content/$directory_name')


# In[ ]:


# Setup trainer
trainer = DetectionModelTrainer()  
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=output)
trainer.setTrainConfig(object_names_array=['plastic', 'metal', 'paper', 'glass'],
                       batch_size=4, num_experiments=20,
                       train_from_pretrained_model="pretrained-yolov3.h5"
                       )
trainer.trainModel()


# In[ ]:


# Image to test
image_path = f'/content/{directory_name}/test/images'
image_name = os.listdir(image_path)[0]
# Minimum detection confidence threshold percentage. If you don't get any detections, try lowering the threshold. 
min_threshold = 30

# Get our model from google colab
get_ipython().system('cp -r /content/drive/My\\ Drive/$model_folder/* /content/$directory_name/models')

# Get the latest model saved (with the lowest loss)
all_models = os.listdir(f'/content/{directory_name}/models')
all_models.sort()
best_model = all_models[-1]

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(f"/content/{directory_name}/models/{best_model}")
detector.setJsonPath(f"/content/{directory_name}/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=f"{image_path}/{image_name}",
                                             output_image_path="/content/detections.jpg",
                                             minimum_percentage_probability=min_threshold)

# Show image with detections
image = PIL.Image.open("/content/detections.jpg")
display(image)


# In[ ]:


# Save best model to colab
get_ipython().system('mkdir /content/drive/My\\ Drive/{model_folder}')
get_ipython().system('cp /content/{directory_name}/models/{best_model} /content/drive/My\\ Drive/{model_folder}/{best_model}')
get_ipython().system('cp /content/{directory_name}/json/detection_config.json /content/drive/My\\ Drive/{model_folder}')
