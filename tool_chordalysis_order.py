import numpy as np
import scipy.stats as sts
import pandas
import pandas as pd
import eif_old as eif_old_class
import scipy.io as sio
import os
import sklearn.metrics as skm



text = "[gill-attachment veil-color edible ][bruises gill-attachment gill-spacing stalk-shape edible ][cap-surface gill-spacing gill-size stalk-shape edible ][bruises gill-size stalk-shape stalk-surface-above-ring edible ][bruises gill-size stalk-shape habitat edible ][bruises odor gill-size stalk-shape edible ][bruises gill-size stalk-shape stalk-color-above-ring edible ][bruises gill-size stalk-shape stalk-color-below-ring edible ][bruises gill-size stalk-shape spore-print-color edible ][cap-color bruises gill-size stalk-shape edible ][bruises gill-size gill-color stalk-shape edible ][cap-shape gill-size stalk-shape stalk-root edible ][bruises gill-size stalk-shape stalk-root stalk-surface-below-ring edible ][bruises gill-size stalk-shape stalk-root ring-type edible ][bruises gill-spacing gill-size stalk-shape ring-number edible ][bruises gill-size stalk-shape stalk-root ring-number edible ][bruises gill-spacing stalk-shape ring-number population edible ]"
text = text.replace("[","")
text = text.replace("]","")
text = text.split(" ")
print(text)

new_text = []
for t in text:
    if t != "":
        if t in new_text:
            True
        else:
            new_text.append(t)

print(new_text)


