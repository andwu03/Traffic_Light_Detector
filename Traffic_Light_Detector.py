# !pip install -Uqq fastbook
# !pip install fastai --upgrade
#necessary pip installs

import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.all import *
from fastai.vision.widgets import *

learn_inf = load_learner("model.pkl")

btn_run = widgets.Button(description='Classify')
btn_run

def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)

btn_upload = widgets.FileUpload()

VBox([widgets.Label('Select an image to be classified: '), 
      btn_upload, btn_run, out_pl, lbl_pred])

#Code used to train the model
'''
key = os.environ.get('AZURE_SEARCH_KEY', "insert your search key here")

light_types = "green", "yellow", "red"

path = Path("traffic_lights")

if not path.exists():
  path.mkdir()
  for type in light_types:
    dest = (path/type)
    dest.mkdir(exist_ok=True)
    results = search_images_bing(key, f'{type} traffic light')
    download_images(dest, urls=results.attrgot('contentUrl'))

fns = get_image_files(path)

failed_images = verify_images(fns)

failed_images.map(Path.unlink)

traffic_light = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(224))

traffic_light = traffic_light.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = traffic_light.dataloaders(path)

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(10)
#train for 10 epochs

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

learn.export()
path = Path()
path.ls(file_exts=".pkl")
#file name will be export.pkl, you can then rename it to what you want it to be
'''