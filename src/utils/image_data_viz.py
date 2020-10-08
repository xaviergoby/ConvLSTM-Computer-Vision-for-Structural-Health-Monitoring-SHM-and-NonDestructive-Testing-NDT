import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
from src.utils.data_preprocessing import normalize_img



img1 = cv2.imread(r"C:\Users\Xavier\PycharmProjects\VideoClassificationAndVisualNavigationViaRepresentationLearning\research_and_dev\slam\altered_mm_img_52_52_45.PNG", 0)


def viz_img(img, ):
	"""
	:param img: img array or img file path
	:return:
	"""
	if isinstance(img, str) is True:
		img = cv2.imread(r"{0}".format(img), 0)
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
		
if __name__ == "__main__":
	# cv2.IMREAD_UNCHANGED	w/ flag: -1
	# cv2.IMREAD_GRAYSCALE	 w/ flag: 0
	# cv2.IMREAD_COLOR	 w/ flag: 1 DEFAULT FLAG (BGR color format)
	# cv2.COLOR_BGR2RGB
	img_path = r"C:\Users\Xavier\LSTMforSHM\data\image_datasets\scenario_1\test\0\Cr_0%_1.png"
	# raw_img_pre_normed = cv2.imread(img_path)
	gray_pre_normed = cv2.imread(img_path, 0)
	normed_img = normalize_img(gray_pre_normed)
	boundary_edge = np.ones((247, 50), dtype=np.uint8)*255
	gray_pre_normed_lhs_crop = gray_pre_normed[:, :2051]
	normed_img_lhs_crop = normed_img[:, 2051:]
	pre_and_post_normed_gray_scale_img = cv2.hconcat([gray_pre_normed, boundary_edge, normed_img])
	cv2.namedWindow('Pre and Post Normalization Gray Scale Image', cv2.WINDOW_NORMAL)
	cv2.imshow('Pre and Post Normalization Gray Scale Image', pre_and_post_normed_gray_scale_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()