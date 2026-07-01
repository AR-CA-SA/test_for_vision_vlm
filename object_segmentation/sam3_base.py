from ultralytics.models.sam import SAM3VideoSemanticPredictor


overrides = dict(conf=0.25, task="segment", mode="predict" , model="sam3.pt" ,  save=True)

model_sam_3 = SAM3VideoSemanticPredictor(overrides=overrides)
