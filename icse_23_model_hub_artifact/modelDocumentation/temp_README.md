---
language: en
license: apache-2.0
tags:
- generated_from_trainer
- image-classification
datasets:
- beans
metrics:
- accuracy
widget:
- src: https://huggingface.co/nateraw/vit-base-beans/resolve/main/healthy.jpeg
  example_title: Healthy
- src: https://huggingface.co/nateraw/vit-base-beans/resolve/main/angular_leaf_spot.jpeg
  example_title: Angular Leaf Spot
- src: https://huggingface.co/nateraw/vit-base-beans/resolve/main/bean_rust.jpeg
  example_title: Bean Rust
model-index:
- name: vit-base-beans
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      type: beans
      name: beans
      args: default
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9774436090225563
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      name: beans
      type: beans
      config: default
      split: test
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9453125
      verified: true
    - name: Precision Macro
      type: precision
      value: 0.9453325082933705
      verified: true
    - name: Precision Micro
      type: precision
      value: 0.9453125
      verified: true
    - name: Precision Weighted
      type: precision
      value: 0.9452605321507761
      verified: true
    - name: Recall Macro
      type: recall
      value: 0.945736434108527
      verified: true
    - name: Recall Micro
      type: recall
      value: 0.9453125
      verified: true
    - name: Recall Weighted
      type: recall
      value: 0.9453125
      verified: true
    - name: F1 Macro
      type: f1
      value: 0.9451827242524917
      verified: true
    - name: F1 Micro
      type: f1
      value: 0.9453125
      verified: true
    - name: F1 Weighted
      type: f1
      value: 0.944936150332226
      verified: true
    - name: loss
      type: loss
      value: 0.26030588150024414
      verified: true
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# vit-base-beans

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the beans dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0942
- Accuracy: 0.9774

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 1337
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.2809        | 1.0   | 130  | 0.2287          | 0.9699   |
| 0.1097        | 2.0   | 260  | 0.1676          | 0.9624   |
| 0.1027        | 3.0   | 390  | 0.0942          | 0.9774   |
| 0.0923        | 4.0   | 520  | 0.1104          | 0.9699   |
| 0.1726        | 5.0   | 650  | 0.1030          | 0.9699   |


### Framework versions

- Transformers 4.10.0.dev0
- Pytorch 1.9.0+cu102
- Datasets 1.11.1.dev0
- Tokenizers 0.10.3
