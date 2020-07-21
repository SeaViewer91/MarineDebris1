from darkflow.net.build import TFNet

# Training using custom images
options = {
	'model' : 'cfg/MD_2/yolo_SP1.cfg',
	'load' : 'bin/yolo.weights',
	'trainer' : 'adam',
	'batch' : 5,
	'epoch' : 3,
	'train' : True,
	'lr' : 1e-3,
	# 'save' : 2000,
	'keep' : 3,
	'gpu' : 0.7,
	'annotation' : 'annotation5/',
	'dataset' : 'images5/'
}

tfnet = TFNet(options)

tfnet.train()
