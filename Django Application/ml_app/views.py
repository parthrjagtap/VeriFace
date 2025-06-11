from django.shortcuts import render, redirect
import torch
import torchvision
from torchvision import transforms, models
from torch.nn.functional import softmax

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
import json
import glob
import copy
from torchvision import models
import shutil
from PIL import Image as pImage
import time
from django.conf import settings
from .forms import VideoUploadForm

index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
if torch.cuda.is_available():
    device = 'gpu'
else:
    device = 'cpu'

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

class Model(nn.Module):

    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))


class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length=60,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        for i,frame in enumerate(self.frame_extract(video_path)):
            #if(i % a == first_frame):
            faces = face_recognition.face_locations(frame)
            try:
              top,right,bottom,left = faces[0]
              frame = frame[top:bottom,left:right,:]
            except:
              pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        """
        for i,frame in enumerate(self.frame_extract(video_path)):
            if(i % a == first_frame):
                frames.append(self.transform(frame))
        """        
        # if(len(frames)<self.count):
        #   for i in range(self.count-len(frames)):
        #         frames.append(self.transform(frame))
        #print("no of frames", self.count)
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image

def im_convert(tensor, video_file_name):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    # This image is not used
    # cv2.imwrite(os.path.join(settings.PROJECT_DIR, 'uploaded_images', video_file_name+'_convert_2.png'),image*255)
    return image

def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype('uint8'))
    plt.show()


def predict(model,img,path = './', video_file_name=""):
  fmap,logits = model(img.to(device))
  img = im_convert(img[:,-1,:,:,:], video_file_name)
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  confidence = logits[:,int(prediction.item())].item()*100
  print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)  
  return [int(prediction.item()),confidence]

# def plot_heat_map(i, model, img, path = './', video_file_name=''):
#   fmap,logits = model(img.to(device))
#   params = list(model.parameters())
#   weight_softmax = model.linear1.weight.detach().cpu().numpy()
#   logits = sm(logits)
#   _,prediction = torch.max(logits,1)
#   idx = np.argmax(logits.detach().cpu().numpy())
#   bz, nc, h, w = fmap.shape
#   #out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
#   out = np.dot(fmap[i].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
#   predict = out.reshape(h,w)
#   predict = predict - np.min(predict)
#   predict_img = predict / np.max(predict)
#   predict_img = np.uint8(255*predict_img)
#   out = cv2.resize(predict_img, (im_size,im_size))
#   heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
#   img = im_convert(img[:,-1,:,:,:], video_file_name)
#   result = heatmap * 0.5 + img*0.8*255
#   # Saving heatmap - Start
#   heatmap_name = video_file_name+"_heatmap_"+str(i)+".png"
#   image_name = os.path.join(settings.PROJECT_DIR, 'uploaded_images', heatmap_name)
#   cv2.imwrite(image_name,result)
#   # Saving heatmap - End
#   result1 = heatmap * 0.5/255 + img*0.8
#   r,g,b = cv2.split(result1)
#   result1 = cv2.merge((r,g,b))
#   return image_name

# Model Selection


#

def plot_heat_map(i, model, img, path='./', video_file_name='', max_heatmaps=15):
    """
    Fast CAM for a single frame index `i`, but reuses a single forward pass for all frames.
    `img` is expected of shape [1, S, C, H, W].
    Limits the number of heatmaps to `max_heatmaps` for sequences longer than that.
    """
    device = next(model.parameters()).device

    # 1) Forward pass on the whole batch once
    fmap, logits = model(img.to(device))  # fmap: [1*S, C, fh, fw], logits: [1, num_classes]
    # Bring everything to CPU / numpy once
    fmap_np = fmap.detach().cpu().numpy()  # shape [S, C, fh, fw]
    probs = softmax(logits, dim=1).detach().cpu().numpy()[0]  # [num_classes]
    weights = model.linear1.weight.detach().cpu().numpy()  # [num_classes, C]

    # 2) Pull out the single frame’s feature-map & compute CAM
    S, C, fh, fw = fmap_np.shape  # fmap_np is [S, C, fh, fw]
    class_idx = np.argmax(probs)
    cam_w = weights[class_idx]  # [C]

    # Iterate only over a limited number of frames
    limit = min(S, max_heatmaps)  # Ensure we don't exceed max_heatmaps even if the sequence is longer

    heatmap_names = []
    for frame_idx in range(limit):  # Loop over the limited number of frames
        # Feature map for frame `frame_idx`
        fm = fmap_np[frame_idx].reshape(C, fh * fw)  # [C, fh*fw]
        cam = np.dot(cam_w, fm).reshape(fh, fw)  # [fh, fw]

        # 3) Normalize and convert to 0–255
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam = np.uint8(255 * cam)

        # 4) Resize & colorize
        cam_resized = cv2.resize(cam, (im_size, im_size), interpolation=cv2.INTER_NEAREST)
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

        # 5) Overlay on original frame
        orig = im_convert(img[0, frame_idx].unsqueeze(0), video_file_name)  # H×W×3 float [0,1]
        overlay = np.clip(heatmap * 0.5 + orig * 0.8 * 255, 0, 255).astype('uint8')

        # 6) Save and return path
        heatmap_name = f"{video_file_name}_heatmap_{frame_idx}.png"
        save_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', heatmap_name)
        cv2.imwrite(save_path, overlay)
        heatmap_names.append(heatmap_name)

    return heatmap_names

def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))

    for model_path in list_models:
        model_name.append(os.path.basename(model_path))

    for model_filename in model_name:
        try:
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except IndexError:
            pass  # Handle cases where the filename format doesn't match expected

    if len(sequence_model) > 1:
        accuracy = []
        for filename in sequence_model:
            acc = filename.split("_")[1]
            accuracy.append(acc)  # Convert accuracy to float for proper comparison
        max_index = accuracy.index(max(accuracy))
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[max_index])
    elif len(sequence_model) == 1:
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[0])
    else:
        print("No model found for the specified sequence length.")  # Handle no models found case

    return final_model

ALLOWED_VIDEO_EXTENSIONS = set(['mp4','gif','webm','avi','3gp','wmv','flv','mkv'])

def allowed_video_file(filename):
    #print("filename" ,filename.rsplit('.',1)[1].lower())
    if (filename.rsplit('.',1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS):
        return True
    else: 
        return False
def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})

            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})
            
            if allowed_video_file(video_file.name) == False:
                video_upload_form.add_error("upload_video_file","Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            
            saved_video_file = 'uploaded_file_'+str(int(time.time()))+"."+video_file_ext
            if settings.DEBUG:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            else:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file)
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

def predict_page(request):
    predict_template_name = 'predict.html'
    if request.method == "GET":
        # Redirect to 'home' if 'file_name' is not in session
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        if 'file_name' in request.session:
            video_file = request.session['file_name']
        if 'sequence_length' in request.session:
            sequence_length = request.session['sequence_length']
        path_to_videos = [video_file]
        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        # Production environment adjustments
        if not settings.DEBUG:
            production_video_name = os.path.join('/home/app/staticfiles/', video_file_name.split('/')[3])
            print("Production file name", production_video_name)
        else:
            production_video_name = video_file_name

        # Load validation dataset
        video_dataset = validation_dataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)

        # Load model
        if(device == "gpu"):
            model = Model(2).cuda()  # Adjust the model instantiation according to your model structure
        else:
            model = Model(2).cpu()  # Adjust the model instantiation according to your model structure
        model_name = os.path.join(settings.PROJECT_DIR, 'models', get_accurate_model(sequence_length))
        path_to_model = os.path.join(settings.PROJECT_DIR, model_name)
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()
        start_time = time.time()
        # Display preprocessing images
        print("<=== | Started Videos Splitting | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()

        print(f"Number of frames: {len(frames)}")
        # Process each frame for preprocessing and face cropping
        padding = 40
        faces_found = 0
        for i in range(sequence_length):
            if i >= len(frames):
                break
            frame = frames[i]

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save preprocessed image
            image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_rgb = pImage.fromarray(rgb_frame, 'RGB')
            img_rgb.save(image_path)
            preprocessed_images.append(image_name)

            # Face detection and cropping
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 0:
                continue

            top, right, bottom, left = face_locations[0]
            frame_face = frame[top - padding:bottom + padding, left - padding:right + padding]

            # Convert cropped face image to RGB and save
            rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
            image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_face_rgb.save(image_path)
            faces_found += 1
            faces_cropped_images.append(image_name)

        print("<=== | Videos Splitting and Face Cropping Done | ===>")
        print("--- %s seconds ---" % (time.time() - start_time))

        # No face detected
        if faces_found == 0:
            return render(request, 'predict_template_name.html', {"no_faces": True})

        # Perform prediction
        try:
            heatmap_images = []
            output = ""
            confidence = 0.0

            for i in range(len(path_to_videos)):
                print("<=== | Started Prediction | ===>")
                prediction = predict(model, video_dataset[i], './', video_file_name_only)
                confidence = round(prediction[1], 1)
                output = "REAL" if prediction[0] == 1 else "FAKE"
                print("Prediction:", prediction[0], "==", output, "Confidence:", confidence)
                print("<=== | Prediction Done | ===>")
                print("--- %s seconds ---" % (time.time() - start_time))

                # Uncomment if you want to create heat map images
                heatmap_images = []
                for j in range(7):
                    try:
                        print(f"Generating heatmap for frame {j}...")
                        hm = plot_heat_map(j, model, video_dataset[i], './', video_file_name_only)
                        heatmap_images.append(hm)
                        print(f"Successfully generated heatmap for frame {j}")
                    except Exception as e:
                        print(f"🔥 Heatmap generation failed for frame {j}: {e}")

            # Render results
            context = {
                'preprocessed_images': preprocessed_images,
                'faces_cropped_images': faces_cropped_images,
                'heatmap_images': heatmap_images,  # <— make sure this line is present
                'original_video': production_video_name,
                'output': output,
                'confidence': confidence
            }

            if settings.DEBUG:
                print(">>> Reached render block with context:", context.keys())
                return render(request, predict_template_name, context)
            else:
                print(">>> Reached render block with context:", context.keys())
                return render(request, predict_template_name, context)

        except Exception as e:
            print(f"Exception occurred during prediction: {e}")
            return render(request, 'cuda_full.html')
def about(request):
    return render(request, about_template_name)

def handler404(request,exception):
    return render(request, '404.html', status=404)
def cuda_full(request):
    return render(request, 'cuda_full.html')

def research(request):
    return render(request, 'research.html')

def contact(request):
    return render(request, 'contact.html')


