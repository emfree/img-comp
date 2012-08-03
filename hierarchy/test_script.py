from hierarchy2 import *



predictions = []
images = []


for i in range(20):
    path = "../test-images/test-images-513/%03d-0.png" % (i + 1)
    img = Image.open(path)
    pngsize = os.path.getsize(path) * 8
    pred1 = predict(img, 'mvl')[0]
    pred2 = predict(img, 'mvl2')[0]
    acl1 = int(approx_codelength(pred1)) + 32
    acl2 = int(approx_codelength(pred2)) + 32
    print i + 1, 1. * acl1 / pngsize, 1. * acl2 / pngsize


