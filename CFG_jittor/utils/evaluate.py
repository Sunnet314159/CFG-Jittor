import os
from glob import glob
import numpy as np
from PIL import Image
import torch
import torchvision
from pytorch_fid.fid_score import calculate_fid_given_paths
from scipy.stats import entropy
import tqdm

def evaluate_metrics(gen_dir, real_status_path, num_images=1000):
    '''
    calculate the FID and IS scores for the generated images. 
    Due to the high complexity of the inplementation of jittor, we use the PyTorch implementation instead.
    '''
   
    # FID
    fid = calculate_fid_given_paths(
        [gen_dir, real_status_path], 
        batch_size=50,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048, 
        num_workers=0 
    )
    
    # IS
    inception = torchvision.models.inception_v3(weights='DEFAULT')
    inception.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inception = inception.to(device)    

    def preprocess_images(image_dir, num_imgs_limit):
        transform_ops = torchvision.transforms.Compose([
            torchvision.transforms.Resize((299, 299)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
        ])
        images = []
        image_paths = sorted(glob(os.path.join(image_dir, '*.png')) + glob(os.path.join(image_dir, '*.jpg')) + glob(os.path.join(image_dir, '*.jpeg')))[:num_imgs_limit]
        
        for img_path in tqdm.tqdm(image_paths, desc=f"Preprocessing {len(image_paths)} images"):
            img = Image.open(img_path).convert('RGB')
            img = transform_ops(img)
            images.append(img)
        return torch.stack(images)

    gen_images = preprocess_images(gen_dir, num_images).to(device)
    with torch.no_grad():
        logits = inception(gen_images)
        preds = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
    
    n_split = 10 
    if len(preds) < n_split:
            n_split = len(preds) 

    # 计算 IS
    scores = []
    for i in tqdm.tqdm(range(n_split), desc="Calculating IS Scores"):
        p_yx = preds[i * len(preds) // n_split: (i + 1) * len(preds) // n_split, :]
        p_y = np.mean(p_yx, axis=0) 
        kl_div = np.sum(p_yx * (np.log(p_yx + 1e-10) - np.log(p_y + 1e-10)), axis=1)
        scores.append(np.exp(np.mean(kl_div)))
    is_score = (np.mean(scores), np.std(scores)) 

    return fid, is_score