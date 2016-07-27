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

local reshape = 11

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
	-- 	--Network Type Sequential, Parrallel, etc
		if idx == 1 then
	-- 	--Network Layers
		else
			local supported, n, w, b, wc, bc, wlt, blt, linear_v, conv_v, pool_v = processLayer(value)
			if supported then
				appendBinary(file,n,w,b,wc,bc,wlt,blt,linear_v,conv_v,pool_v)
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
	elseif name == "nn.SpatialMaxPooling" then
		id = pooling_max
	elseif name == "nn.SpatialAveragePooling" then
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
	elseif name == "nn.Reshape" then
		id = reshape
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

function appendBinary(file, name, weight, bias, weight_c, bias_c, weight_layer_type, bias_layer_type, linear_values, conv_values, pool_values)
	--Write Layer Type ID
	local layerid = resolveLayerName(name)
	file:writeInt(layerid)

	if name == "nn.Linear" then
		for i=1,#linear_values do
			print ("here"..linear_values[i])
			file:writeInt(linear_values[i])
		end
	elseif name == "nn.SpatialConvolutionMM" or name == "nn.SpatialConvolution" then
		for i=1,#conv_values do
			file:writeInt(conv_values[i])
		end
	elseif name == "nn.SpatialMaxPooling" or name == "nn.SpatialAveragePooling" then
		for i=1,#pool_values do
			file:writeInt(pool_values[i])
		end
	else
		for i=1,#linear_values do
			print ("here"..linear_values[i])
			file:writeInt(linear_values[i])
		end
	end

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

	local linear_input_size = nil
	local linear_output_size = nil

	local conv_input_plane = nil
	local conv_output_plane = nil
	local conv_kernel_width = nil
	local conv_kernel_height = nil
	local conv_shift_width = nil
	local conv_shift_height = nil
	local conv_pad_width = nil
	local conv_pad_height = nil

	local pool_kernel_width = nil
	local pool_kernel_height = nil
	local pool_shift_width = nil
	local pool_shift_height = nil
	local pool_pad_width = nil
	local pool_pad_height = nil

	if name == "nn.Linear" then
		linear_input_size = layerData.gradInput:size(1)
		linear_output_size = layerData.output:size(1)
	elseif name == "nn.SpatialConvolutionMM" or name == "nn.SpatialConvolution" then
		linear_input_size = layerData.gradInput:size(1)
		linear_output_size = layerData.output:size(1)
		conv_input_plane = layerData.nInputPlane
		conv_output_plane = layerData.nOutputPlane
		conv_kernel_width = layerData.kW
		conv_kernel_height = layerData.kH
		conv_shift_width = layerData.dW
		conv_shift_height = layerData.dH
		conv_pad_width = layerData.padW
		conv_pad_height = layerData.padH
	elseif name == "nn.SpatialMaxPooling" or name == "nn.SpatialAveragePooling" then
		linear_input_size = layerData.gradInput:size(1)
		linear_output_size = layerData.output:size(1)
		pool_kernel_width = layerData.kW
		pool_kernel_height = layerData.kH
		pool_shift_width = layerData.dW
		pool_shift_height = layerData.dH
		pool_pad_width = layerData.padW
		pool_pad_height = layerData.padH
	else
		linear_input_size = layerData.gradInput:size(1)
		linear_output_size = layerData.output:size(1)
	end
	
	if layerData.weight ~= nil then
	 	weight = torch2ios_utils.flatten(layerData.weight)
	 	weight_c = weight:size(1)
	 	wltype = weight:type()
	end
	if layerData.bias ~= nil then
	 	bias = torch2ios_utils.flatten(layerData.bias)
	 	bias_c = bias:size(1)
	 	bltype = bias:type()
	end

	return true, name, weight, bias, weight_c, bias_c, wltype, bltype, {linear_input_size,linear_output_size}, {conv_input_plane,conv_output_plane,conv_kernel_width,conv_kernel_height,conv_shift_width,conv_shift_height,conv_pad_width,conv_pad_height},{pool_kernel_width,pool_kernel_height,pool_shift_width,pool_shift_height,pool_pad_width,pool_pad_height}
end