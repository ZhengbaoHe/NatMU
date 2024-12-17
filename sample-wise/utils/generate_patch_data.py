import numpy as np
from PIL import Image

def generate_mask_cifar(delta):
    pattern_forget_data = np.zeros((4,3,32,32), dtype=np.float32)
    a = 2
    b = 32 // a
    for i in range(a):
        for j in range(b):
            if i%2==0:
                pattern_forget_data[0,: , i*b+j, :]= j/b
                pattern_forget_data[1,: , :, i*b+j]= j/b
                pattern_forget_data[2,: , i*b+j, :]= (b-1)/b - j/b
                pattern_forget_data[3,: , :, i*b+j]= (b-1)/b - j/b
            else:
                
                pattern_forget_data[0,: , i*b+j, :]= (b-1)/b - j/b
                pattern_forget_data[1,: , : ,i*b+j]= (b-1)/b - j/b
                pattern_forget_data[2,: , i*b+j, :]= j/b
                pattern_forget_data[3,: , : ,i*b+j]= j/b
    pattern_forget_data = np.clip((pattern_forget_data + delta), 0, 1)
    print("Mask Info:", pattern_forget_data.sum(axis=(1,2,3)) / np.ones_like(pattern_forget_data, dtype=np.float32).sum(axis=(1,2,3)))
    return pattern_forget_data, (1 - pattern_forget_data)


def generate_mask_tinyimagenet(delta):
    pattern_forget_data = np.zeros((4,3,64,64), dtype=np.float32)
    a = 2
    b = 64 // a
    pattern_forget_data = np.zeros((4,3,64,64), dtype=np.float32)
    for i in range(a):
        for j in range(b):
            if i%2==0:
                pattern_forget_data[0,: , i*b+j, :]= j/b
                pattern_forget_data[1,: , :, i*b+j]= j/b
                pattern_forget_data[2,: , i*b+j, :]= (b-1)/b - j/b
                pattern_forget_data[3,: , :, i*b+j]= (b-1)/b - j/b
            else:
                
                pattern_forget_data[0,: , i*b+j, :]= (b-1)/b - j/b
                pattern_forget_data[1,: , : ,i*b+j]= (b-1)/b - j/b
                pattern_forget_data[2,: , i*b+j, :]= j/b
                pattern_forget_data[3,: , : ,i*b+j]= j/b

    pattern_forget_data = np.clip((pattern_forget_data + delta), 0, 1)
    print("Mask Info:", pattern_forget_data.sum(axis=(1,2,3)) / np.ones_like(pattern_forget_data, dtype=np.float32).sum(axis=(1,2,3)))
    
    return pattern_forget_data, (1 - pattern_forget_data)



def get_random_idx_of_retainData(retain_labels, forget_labels, pattern_length, num_of_classes):
    classes = list(range(num_of_classes))
    tmp_idxs = {}
    count = {}
    for class_ in classes:
        tmp_idx = np.nonzero(retain_labels == class_)[0].copy()
        np.random.shuffle(tmp_idx)
        tmp_idxs[class_] = tmp_idx
        count[class_] = 0
    
    random_idx = []
    for idx in range(len(forget_labels)):
        forget_class = forget_labels[idx]
        classes = list(range(num_of_classes))
        classes.remove(forget_class)
        random_class = np.random.choice(classes)
        tmp_idx = tmp_idxs[random_class]
        start = count[random_class]
        random_idx.append(tmp_idx[start*pattern_length:start*pattern_length+pattern_length])
        count[random_class] += 1
    random_idx = np.concatenate(random_idx, axis=0)
    return random_idx


def get_pure_random_idx_of_retainData(retain_labels, forget_labels, pattern_length, num_of_classes):
    random_idx_of_retainset = np.random.permutation(len(retain_labels))
    forget_label_repeat = np.repeat(forget_labels, pattern_length, axis=0)
    position_eq = retain_labels[random_idx_of_retainset[:len(forget_label_repeat)]] ==  forget_label_repeat
    while position_eq.sum()>0:
        position_neq = np.nonzero(retain_labels[random_idx_of_retainset[:len(forget_label_repeat)]] !=  forget_label_repeat)[0]
        neq_mask = np.isin(np.arange(len(random_idx_of_retainset)), position_neq)
        random_idx_of_retainset[~neq_mask] = np.random.permutation(random_idx_of_retainset[~neq_mask])
        position_eq = retain_labels[random_idx_of_retainset[:len(forget_label_repeat)]] ==  forget_label_repeat
    return random_idx_of_retainset[:len(forget_label_repeat)]




def save_patch_img(patch_data, pattern_length):
    for i in range(len(patch_data)):
        img = patch_data[i].transpose(1,2,0)
        img = Image.fromarray(img.astype(np.uint8))
        img.save(f"imgs/{i//pattern_length}_{i%pattern_length}.png")
        
        
if __name__ == "__main__":
    
    mask = generate_mask_tinyimagenet(0)[0]
    from PIL import Image
    for i in range(4):
        img = np.transpose(mask[i], (1,2,0)) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.save(f"{i}.png")
    