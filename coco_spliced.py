from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os,shutil
import cv2

random.seed(10)

dataDir = 'train2017'
dataType = 'train2017'
annFile = 'instances_{}.json'.format(dataType)

# Initialize the COCO api for instance annotations
coco = COCO(annFile)

# Filter classes
filterClasses = [ 'cell phone']
catIds = coco.getCatIds(catNms=filterClasses)
imgIds = coco.getImgIds(catIds=catIds)

# Destination paths
spliced_path = 'coco/spliced'
mask_path = 'coco/mask'
if os.path.exists(spliced_path):
    shutil.rmtree(spliced_path)
    shutil.rmtree(mask_path)

os.makedirs(spliced_path, exist_ok=True)
os.makedirs(mask_path, exist_ok=True)




# Text file to store paths
with open('coco_spliced_dataset.txt', 'w') as txt_file:

    # Iterate through imagesfrandom
    for c, img_id in enumerate(imgIds):
        img = coco.loadImgs([img_id])[0]
        I = io.imread('{}/{}'.format(dataDir, img['file_name']))
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)  # Convert to RGB

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # Create mask
        mask = np.zeros((img['height'], img['width']))
        for i in range(len(anns)):
            mask = np.maximum(coco.annToMask(anns[i]), mask)

        # Splice region
        spliced_area = I * mask[:, :, None].astype(I.dtype)
        mask_area = mask.astype(I.dtype)  # Work with the whole mask

        # Apply random rotation or flipping
        if random.choice([True, False]):
            angle = random.choice([90, -90])
            M = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), angle, 1)
            spliced_area = cv2.warpAffine(spliced_area, M, (mask.shape[1], mask.shape[0]))
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        else:
            flipCode = random.randint(-1, 1)
            spliced_area = cv2.flip(spliced_area, flipCode=flipCode)
            mask = cv2.flip(mask, flipCode=flipCode)

        # Manipulate 8 random images
        random_images = random.sample([i for i in imgIds if i != img_id], 8)
        for rand_img_id in random_images:
            rand_img = coco.loadImgs([rand_img_id])[0]
            rand_I = io.imread('{}/{}'.format(dataDir, rand_img['file_name']))
            rand_I = cv2.cvtColor(rand_I, cv2.COLOR_BGR2RGB)  # Convert to RGB

            # Resize the spliced area if it's larger than the random image
            scale_factor = min(
                (rand_I.shape[0] - 2) / spliced_area.shape[0],
                (rand_I.shape[1] - 2) / spliced_area.shape[1]
            )

            spliced_area_resized = cv2.resize(spliced_area, (0, 0), fx=scale_factor, fy=scale_factor)
            mask_area_resized = cv2.resize(mask_area, (spliced_area_resized.shape[1], spliced_area_resized.shape[0])) # Resize the mask to match
            

            print(spliced_area_resized.shape , mask_area_resized.shape)

            # Define the position where the spliced area will be inserted
            x_offset = random.randint(0, rand_I.shape[1] - spliced_area_resized.shape[1] - 1)
            y_offset = random.randint(0, rand_I.shape[0] - spliced_area_resized.shape[0] - 1)

            # Insert the spliced area into the random image using the mask
            for i in range(spliced_area_resized.shape[0]):
                for j in range(spliced_area_resized.shape[1]):
                    if mask_area_resized[i, j] > 0:
                        rand_I[y_offset + i, x_offset + j] = spliced_area_resized[i, j]


            full_mask = np.zeros_like(rand_I[:,:,0])

            # Insert the resized mask into the full-size mask at the same offsets
            for i in range(mask_area_resized.shape[0]):
                for j in range(mask_area_resized.shape[1]):
                    full_mask[y_offset + i, x_offset + j] = mask_area_resized[i, j]



            # Save manipulated image and mask
            spliced_filename = os.path.join(spliced_path, '{}_spliced.jpg'.format(rand_img['file_name'][:-4]))
            mask_filename = os.path.join(mask_path, '{}_mask.png'.format(rand_img['file_name'][:-4]))
            #rand_I = cv2.cvtColor(rand_I, cv2.COLOR_RGB2BGR)
            print(rand_I.shape , full_mask.shape)
            print(" ")
            cv2.imwrite(spliced_filename, (rand_I).astype(np.uint8) ) # * 255).astype(np.uint8))
            cv2.imwrite(mask_filename, (full_mask * 255)) #.astype(np.uint8))

            # Write to text file
            txt_file.write('{}, {}\n'.format(spliced_filename, mask_filename))

        # if c == 200:  #to run 200 images 
        #     break
