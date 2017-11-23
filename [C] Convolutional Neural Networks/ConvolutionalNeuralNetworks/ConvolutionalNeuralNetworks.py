from Scripts import ImageHelper,Convolutions
from Enums import Filters

#Read contents of the image
filename = "images/img1.jpg"
image = ImageHelper.get_image(filename)

#Convert to Black and white stream
bnwimage = ImageHelper.convert_to_gray(image)

#using NORMAL filter

vertical_normal = Convolutions.generate_filtered_image(bnwimage, Filters.NORMAL)
ImageHelper.save_image("vertical_normal.jpg",vertical_normal, 'gray') 

#using SOBEL filter

vertical_sobel = Convolutions.generate_filtered_image(bnwimage, Filters.SOBEL)
ImageHelper.save_image("vertical_sobel.jpg",vertical_sobel, 'gray')

#using SCHARR filter

vertical_scharr = Convolutions.generate_filtered_image(bnwimage, Filters.SCHARR)
ImageHelper.save_image("vertical_scharr.jpg",vertical_scharr, 'gray')