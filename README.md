# DreamBoothOnRay
Fine tune dream booth model using Ray Dataset and Ray Train on a single g5.12xlarge instance using 2 A10G GPUs.

The demo tunes both the text_encoder and unet parts of Stable Diffusion, and utilizes the prior preserving loss function.

<p align="center">
  <img src="https://github.com/gjoliver/DreamBoothOnRay/blob/master/assets/example.png" />
</p>

### Overview

First, we download the pre-trained stable diffusion model as a starting point.

We will then train this model with a few images of our subject, for instance our cat.

Remember how the model creates images using a prompt? For instance "A photo of a cat" will create a photo of any cat. But we want the model to output a photo of a specific cat.

To achieve this, we choose a non-word as an identifier, e.g. "sks". When fine-tuning the model with our cat, we will teach it that the prompt is "A photo of a sks cat". 

After fine-tuning we can run inference with this specific prompt. For instance: "A photo of a sks cat on the moon" will create an image of our cat - on the moon.

### Step 0
Prepare some directories and environment variables

```bash
export ORIG_MODEL_HASH="3857c45b7d4e78b3ba0f39d4d7f50a2a05aa23d4"
export ORIG_MODEL_DIR="./model-orig"
export ORIG_MODEL_PATH="$ORIG_ORIG_MODEL_DIR/snapshots/$ORIG_MODEL_HASH"
export TUNED_MODEL_DIR="./model-tuned"
export IMAGES_REG_DIR="./images-reg"
export IMAGES_OWN_DIR="./images-own"
export IMAGES_NEW_DIR="./images-new"

export CLASS_NAME="cat"

mkdir -p $ORIG_MODEL_DIR $TUNED_MODEL_DIR $IMAGES_REG_DIR $IMAGES_OWN_DIR
```

Copy some images for fine-tuning into `$IMAGES_OWN_DIR`.

### Step 1
Download and cache a pre-trained Stable-Diffusion model locally.
Default model and version are ``CompVis/stable-diffusion-v1-4``
at git hash ``3857c45b7d4e78b3ba0f39d4d7f50a2a05aa23d4``.
```
python cache_model.py --model_dir=$ORIG_MODEL_DIR --revision=$ORIG_MODEL_HASH
```
Note that actual model files will be downloaded into
``\<model_dir>\snapshots\<git_hash>\`` directory.

### Step 2
Create a regularization image set for a class of subjects:
```
python run_model.py \
  --model_dir=$ORIG_MODEL_PATH \
  --output_dir=$IMAGES_REG_DIR \
  --prompts="photo of a $CLASS_NAME" \
  --num_samples_per_prompt=200
```

### Step 3
Save a few (4 to 5) images of the subject being fine-tuned
in a local directory. Then launch the training job with:
```
python train.py \
  --model_dir=$ORIG_MODEL_PATH \
  --output_dir=$TUNED_MODEL_PATH \
  --instance_images_dir=$IMAGES_OWN_DIR \
  --instance_prompt="a photo of unqtkn $CLASS_NAME" \
  --class_images_dir=$IMAGES_REG_DIR \
  --class_prompt="a photo of a $CLASS_NAME"
```

### Step 4
Try your model with the same commandline as Step 2, but point
to your own model this time!

```
python run_model.py \
  --model_dir=$TUNED_MODEL_PATH \
  --output_dir=$IMAGES_NEW_DIR \
  --prompts="photo of a unqtkn $CLASS_NAME" \
  --num_samples_per_prompt=20
```