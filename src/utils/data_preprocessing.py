
def get_compatible_img_and_frame_widths(img_width, frame_width):
	"""
	This function is meant for finding the number closest to n and divisible by m. Within the context
	of data postprocessing of images, this function is used for determining the maximum acceptable frame_width
	of an image(es) (img_width) for creating frames for the image(es) of constant frame_width (frame_width).
	gcd: greatest common divisor
	:param img_width: pixel frame_width (AKA milliseconds) of an image
	:param frame_width: pixel frame_width (AKA milliseconds) of a frame
	:return:
	"""
	q = int(img_width / frame_width)
	n1 = frame_width * q
	if ((img_width * frame_width) > 0):
		n2 = (frame_width * (q + 1))
	else:
		n2 = (frame_width * (q - 1))
	if (abs(img_width - n1) < abs(img_width - n2)):
		return n1
	return n2


#if __name__ == "__main__":
	#img_width = 4101;
	#frame_width = 25
	#print(get_compatible_img_and_frame_widths(img_width, frame_width))
