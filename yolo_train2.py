from darkflow.net.build import TFNet

# Training using custom images
options = {
	'model' : 'cfg/MD_2/yolo_SP1.cfg',
	'load' : 125000,
	'trainer' : 'adam',
	'batch' : 5,
	'epoch' : 30,
	'train' : True,
	'lr' : 1e-5,
	'save' : 5000,
	'keep' : 3,
	'gpu' : 0.7,
	'annotation' : 'annotation/',
	'dataset' : 'images/',
	'backup' : 'ckpt/SP1/'
}

tfnet = TFNet(options)

tfnet.load_from_ckpt()

tfnet.train()
