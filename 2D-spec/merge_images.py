from PIL import Image

s = '6'
img = Image.open("/path/to/shap_figures.png")

img = img.convert("RGBA")
datas = img.getdata()
print(img.size)
thres = 230
newData = []
for item in datas:
    if item[0] >= thres and item[1]>= thres and item[2] >= thres:

        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)
img.putdata(newData)

card = Image.open("/path/to/grey_spec.png").convert("RGBA")

print(card.size)
x, y = card.size
card.paste(img, (0, 0, x, y), img)
card.save("A0" + s + "_spec_gray.png", format="png")