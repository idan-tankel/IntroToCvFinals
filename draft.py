# we will use PIL for that
from PIL import Image,ImageFilter
image = Image.open(r'/home/azureuser/cloudfiles/code/Users/209056712/IntroToCvFinals/new_bnw.jpg')
image_filter = image.filter(ImageFilter.GaussianBlur)
image.show()
image_filter.show()