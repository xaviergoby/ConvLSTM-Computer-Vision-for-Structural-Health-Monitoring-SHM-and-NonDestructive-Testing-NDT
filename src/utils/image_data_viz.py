import matplotlib.pyplot as plt



def viz_img(img):
	num_img_array_dims = len(img.shape)
	plt.figure()
	if num_img_array_dims == 3:
		img_colour_channel = img.shape[2]
		if img_colour_channel == 1:
			plt.figure()
			plt.imshow(img[:, :, 0], cmap='gray')
			plt.show()  # display it
		elif img_colour_channel == 3:
			plt.imshow(img)
			plt.show()
	elif num_img_array_dims == 2:
		plt.imshow(img, cmap="gray")
		plt.show()