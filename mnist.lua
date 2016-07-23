-- Kurt Jacobs
-- RandomDudes
-- 2016

require 'torch'
require 'nn'
require 'nnx'
require 'torch2ios'

torch.setdefaulttensortype('torch.FloatTensor')
mnist_model = torch.load("trainedmnist.net")

saveForiOS(mnist_model, "mnist_ios")