import open_clip
import torch
import torch.nn.functional as F

import tqdm
from torchvision.datasets import STL10


model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

labels = [
    "an airplane",
    "a bird",
    "a car",
    "a cat",
    "a deer",
    "a dog",
    "a horse",
    "a monkey",
    "a ship",
    "a truck"
]

text = tokenizer(labels)
text_embeddings = model.encode_text(text)



def embeddings_to_class_probs(vision_embeddings, text_embeddings):
    vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    logits = vision_embeddings @ text_embeddings.T
    class_probs = F.softmax(100. * logits, dim=-1)
    return class_probs


dataset = STL10(
    root='/Data/federated_learning/kd_jetson_clip_ex/data/',
    download=True,
    split="test"
)

num_correct = 0

for image, label in tqdm.tqdm(dataset):
    input_tensor = preprocess(image).unsqueeze(0)
    vision_embeddings = model.encode_image(input_tensor)
    output_class_probs = embeddings_to_class_probs(vision_embeddings, text_embeddings)
    output_label = torch.argmax(output_class_probs, dim=-1)
    num_correct += int(torch.count_nonzero(output_label == label))

accuracy = 100. * num_correct / len(dataset)
print('\nAccuracy: ', accuracy)

dataset_load =



