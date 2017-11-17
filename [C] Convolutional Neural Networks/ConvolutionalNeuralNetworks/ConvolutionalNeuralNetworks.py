from HandyScripts import ImageHelper

#Read contents of the image
filename = "images/img1.jpg"
image = ImageHelper.get_image(filename)

#Convert to Black and white stream
bnwimage = ImageHelper.convert_to_gray(image)

ImageHelper.show_image(bnwimage, 'gray')