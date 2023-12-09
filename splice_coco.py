
dataDir = 'train2017'
dataType = 'train2017'
annFile = 'instances_{}.json'.format(dataType)
# Destination paths
spliced_path = 'spliced'
mask_path = 'mask'



from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os,shutil,time
import cv2,pdb

random.seed(10)


# Initialize the COCO api for instance annotations
coco = COCO(annFile)

imgIds = coco.getImgIds()


if os.path.exists(spliced_path):
    shutil.rmtree(spliced_path)
    shutil.rmtree(mask_path)

os.makedirs(spliced_path, exist_ok=True)
os.makedirs(mask_path, exist_ok=True)

temp = imgIds.copy()
count = 0
skipped = 0
# Text file to store paths
with open('coco_spliced_800k_new.txt', 'w') as txt_file:


    for w, img_id in enumerate(imgIds):
        img = coco.loadImgs(img_id)[0]
        img_path = os.path.join(dataDir, img['file_name'])
        I = io.imread(img_path)
        
        # Get the corresponding annotation IDs
        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # Create an empty mask
        mask = np.zeros((img['height'], img['width']))
        

        for q in range(len(anns)):
            mask = np.maximum(coco.annToMask(anns[q]), mask)
        

        # Splice region
        try:
            spliced_area = I * mask[:, :, None].astype(I.dtype)
            spliced_area = cv2.cvtColor(spliced_area, cv2.COLOR_RGB2BGR) 
        except:
            skipped+=1
            continue

        # io.imshow(spliced_area)
        # io.show()

        

        # Apply random rotation or flipping
        if random.choice([True, False]):
            angle = random.choice([90, 45, 60, 135, 30, 120, 165])
            M = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), angle, 1)
            spliced_area = cv2.warpAffine(spliced_area, M, (mask.shape[1], mask.shape[0]))
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        else:
            flipCode = random.randint(-1, 1)
            spliced_area = cv2.flip(spliced_area, flipCode=flipCode)
            mask = cv2.flip(mask, flipCode=flipCode)

        mask= np.where(mask < 0.5, 0, 1)
        mask_area = mask.astype(I.dtype)  # Work with the whole mask

        if len(temp)< 10: 
            print("got empty")
            temp = imgIds.copy()
        # Manipulate 8 random images
        random_images = random.sample([i for i in temp if i !=img_id], 8)

        for t,rand_img_id in enumerate(random_images):
            rand_img = coco.loadImgs([rand_img_id])[0]
            rand_I = io.imread('{}/{}'.format(dataDir, rand_img['file_name']))
            rand_I = cv2.cvtColor(rand_I, cv2.COLOR_RGB2BGR)  # Convert to RGB

            # Resize the spliced area if it's larger than the random image
            scale_factor = min(
                (rand_I.shape[0] - 2) / spliced_area.shape[0],
                (rand_I.shape[1] - 2) / spliced_area.shape[1]
            )

            spliced_area_resized = cv2.resize(spliced_area, (0, 0), fx=scale_factor, fy=scale_factor)
            # io.imshow(spliced_area_resized)
            # io.show()
            mask_area_resized = cv2.resize(mask_area, (spliced_area_resized.shape[1], spliced_area_resized.shape[0])) # Resize the mask to match
            #mask_area_resized = mask_area_resized.astype(np.uint8) / 255.0

            # Define the position where the spliced area will be inserted
            x_offset = random.randint(0, rand_I.shape[1] - spliced_area_resized.shape[1] - 1)
            y_offset = random.randint(0, rand_I.shape[0] - spliced_area_resized.shape[0] - 1)
            
            full_mask = np.zeros_like(rand_I[:,:,0])
            full_mask[y_offset:y_offset + mask_area_resized.shape[0], x_offset:x_offset + mask_area_resized.shape[1]] = mask_area_resized
            
           
            # Insert the spliced area into the random image using the mask
            roi = rand_I[y_offset:y_offset + spliced_area_resized.shape[0], x_offset:x_offset + spliced_area_resized.shape[1]]
            
            # Blend the spliced area with the ROI based on the mask
            blended_roi = (spliced_area_resized * mask_area_resized[:, :, None] + roi * (1 - mask_area_resized[:, :, None])).astype(roi.dtype)

            # Assign the blended region back to the target image
            rand_I[y_offset:y_offset + spliced_area_resized.shape[0], x_offset:x_offset + spliced_area_resized.shape[1]] = blended_roi

            # Save manipulated image and mask
            spliced_filename = os.path.join(spliced_path, '{}_spliced_{}.jpg'.format(rand_img['file_name'][:-4], str(count+1)))
            mask_filename = os.path.join(mask_path, '{}_mask_{}.png'.format(rand_img['file_name'][:-4], str(count+1)))
            #rand_I = cv2.cvtColor(rand_I, cv2.COLOR_RGB2BGR)
            unique_values, counts = np.unique(full_mask, return_counts=True)

            if len(counts)==1 or any(count < 10 for count in counts):
                continue
            #rand_I = cv2.cvtColor(rand_I, cv2.COLOR_RGB2BGR)
            cv2.imwrite(spliced_filename, rand_I) # (rand_I).astype(np.uint8) ) # * 255).astype(np.uint8))
            cv2.imwrite(mask_filename, (full_mask * 255)) #.astype(np.uint8))

            # Write to text file
            #txt_file.write('{}, {}\n'.format(spliced_filename.split("/forgery/")[-1], mask_filename).split("/forgery/")[-1])
            txt_file.write('{}, {}\n'.format(spliced_filename.split("/forgery/")[-1], mask_filename.split("/forgery/")[-1]))

            count+=1
            temp.remove(rand_img_id)

            if (count % 1000 == 0):
                print("done ", count, " skipped", skipped)
                
    

print("total spliced iamges = ", count)
