from model_utils import *
import os as os
import sys

def main():
	print(len(sys.argv))
	if len(sys.argv) != 3 and len(sys.argv) != 4:
		raise RuntimeError("only two (input + style) or three arguments " + 
			"(input + style + output) please")
	if not os.path.exists(sys.argv[1]):
		raise RuntimeError("input image not found.")
	if not os.path.exists(sys.argv[2]):
		raise RuntimeError("style image not found.")
	if len(sys.argv) == 3:
		out_path = None
	else:
		out_path = sys.argv[3]

	sess = get_session()
	SAVE_PATH = 'squeezenet.ckpt'
	model = SqueezeNet(save_path=SAVE_PATH, sess=sess)


	"""
    You are welcomed to fiddle with the parameters here
    Params:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss; the larger, the closer output is to the input
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers; the larger, the heavier
    	learned style.
    - tv_weight: weight of total variation regularization term. The larger, the closer neighboring 
    	pixels are, the blurrier/smoother the image. 
    - model: squeeze net model
    - sess: a tenserflow session
    - out_path: output path for the resulting image. If empty, the output goes to output folder, with 
    	the name being input filename + "_out.jpg".
    """

	params = {
	    'content_image' : sys.argv[1],
	    'style_image' : sys.argv[2],
	    'image_size' : 256,
	    'style_size' : 512,
	    'content_layer' : 3,
	    'content_weight' : 5e-2, 
	    'style_layers' : (1, 4, 6, 7),
	    'style_weights' : (20000, 500, 12, 1),
	    'tv_weight' : 5e-2,
	    'model' : model,
	    'sess' : sess,
	    'out_path' : out_path
	}

	style_transfer(**params)

if __name__ == '__main__':
	main()
