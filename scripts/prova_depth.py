import depth_estimator as de
import torchvision
import cv2
import numpy as np

from PIL import Image
img_path = "/scratch/fborgna/NeuralDiff/results/rel/P01_01/decomposition/foreground/foreground_52.png"

#img = torchvision.io.read_image(img_path)
img = cv2.imread(img_path)

depth_est = de.Inference()
depth = depth_est.depth_extractor(img,"niente")

im = Image.fromarray(((depth/np.max(depth)))*255)
im = im.convert("L")
im.save("depth_prova_mia.png")
prova = "ciao"