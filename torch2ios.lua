-- Kurt Jacobs
-- RandomDudes
-- 2016

require 'torch2ios_utils'
require 'torch'

local linear = 1
local spatial_conv = 2
local pooling_max = 3
local pooling_average = 4

local tanh_activation = 5
local hard_tanh_activation = 6
local log_sigmoid_activation = 7
local log_soft_max_activation = 8
local sigmoid_activation = 9
local relu_activation = 10

local torchFloatT = 1
local torchDoubleT = 2
local torchIntT = 3

function saveForiOS(container, filename)
	local file = torch.DiskFile(filename..".t7ios", 'w')
	file:binary()
	local modulesCount = container:listModules()
	--Layer Count
	file:writeInt(#modulesCount-1)
	for idx, value in pairs (modulesCount) do
		--Network Type Sequential, Parrallel, etc
		if idx == 1 then
		--Network Layers
		else
			local supported, n, w, b, wc, bc, wlt, blt = processLayer(value)
			if supported then
				appendBinary(file,n,w,b,wc,bc,wlt,blt)
			else
				print ("unsupported layer with name: "..n.." encountered.")
				file:close()
				os.remove(filename..".t7ios")
				os.exit(-1)
			end
		end
	end
	file:close()
end

function resolveLayerName(name)
	local id = 0
	if name == "nn.Linear" then
		id = linear
	elseif name == "nn.MaxPooling" then
		id = pooling_max
	elseif name == "nn.AveragePooling" then
		id = pooling_average
	elseif name == "nn.SpatialConvolution" or name == "nn.SpatialConvolutionMM" then
		id = spatial_conv
	elseif name == "nn.Tanh" then
		id = tanh_activation
	elseif name == "nn.ReLU" then
		id = relu_activation
	elseif name == "nn.Sigmoid" then
		id = sigmoid_activation
	elseif name == "nn.HardTanh" then
		id = hard_tanh_activation
	elseif name == "nn.LogSigmoid" then
		id = log_sigmoid_activation
	elseif name == "nn.LogSoftMax" then
		id = log_soft_max_activation
	else
		id = -1
	end
	return id
end

function resolveTensorType(tensorType)
	local id = 0
	if tensorType == "torch.FloatTensor" then
		id = torchFloatT
	elseif tensorType == "torch.DoubleTensor" then
		id = torchDoubleT
	elseif tensorType == "torch.IntTensor" then
		id = torchIntT
	end
	return id;
end

function appendBinary(file, name, weight, bias, weight_c, bias_c, weight_layer_type, bias_layer_type)
	--Write Layer Type ID
	local layerid = resolveLayerName(name)
	file:writeInt(layerid)

	--Write Weights
	if weight:nDimension() > 0 then
		local weightTensorTypeID = resolveTensorType(weight_layer_type)
		file:writeInt(weightTensorTypeID)
		file:writeInt(weight_c)
		local t_data = torch.data(weight)
		if weightTensorTypeID == torchFloatT then
			for i = 0,weight:nElement()-1 do file:writeFloat(t_data[i]) end
		elseif weightTensorTypeID == torchIntT then
			for i = 0,weight:nElement()-1 do file:writeInt(t_data[i]) end
		elseif weightTensorTypeID == torchDoubleT then
			for i = 0,weight:nElement()-1 do file:writeDouble(t_data[i]) end
		end
	end

	-- Write Biases
	if bias:nDimension() > 0 then
		local biasTensorTypeID = resolveTensorType(bias_layer_type)
		file:writeInt(biasTensorTypeID)
		file:writeInt(bias_c)
		local t_data = torch.data(bias)
		if biasTensorTypeID == torchFloatT then
			for i = 0,bias:nElement()-1 do file:writeFloat(t_data[i]) end
		elseif weightTensorTypeID == torchIntT then
			for i = 0,bias:nElement()-1 do file:writeInt(t_data[i]) end
		elseif weightTensorTypeID == torchDoubleT then
			for i = 0,bias:nElement()-1 do file:writeDouble(t_data[i]) end
		end
	end
end

function isSupportedLayer(layerName)
	if resolveLayerName(layerName) > 0 then
		return true
	end
	return false
end

function processLayer(layerData)
	--Layer Name
	local name = torch.type(layerData)
	if isSupportedLayer(name) == false then
		return false, name
	end
	 --Layer Weights & Biases
	local weight_c = 0
	local weight = torch.Tensor()
	local bias_c = 0
	local bias = torch.Tensor()

	--Layer Data Type
	local wltype = nil
	local bltype = nil

	if layerData.weight ~= nil then
	 	weight = torch2ios_utils.flatten(layerData.weight)
	 	weight_c = weight:size(1)
	 	wltype = weight:type()
	end
	if layerData.weight ~= nil then
	 	bias = torch2ios_utils.flatten(layerData.bias)
	 	bias_c = bias:size(1)
	 	bltype = bias:type()
	end

	return true, name, weight, bias, weight_c, bias_c, wltype, bltype
end