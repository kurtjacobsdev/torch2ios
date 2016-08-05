require 'torch2ios_utils'

local test_set = torch.load("test_32x32.t7", 'ascii')
local data = test_set.data:type(torch.getdefaulttensortype())
local labels = test_set.labels

-- Normalise Data
local std = std_ or data:std()
local mean = mean_ or data:mean()
data:add(-mean)
data:mul(1/std)

local file = torch.DiskFile("mnist_samples"..".t7iosb", 'w')
file:binary()

-- Extract an arbitrary number of samples [say 5 maybe?]
file:writeInt(5)
for i=1,5,1 do
	d_sample = torch2ios_utils.flatten(data[i])
	d_sample_len = d_sample:size(1)
	l_sample = labels[i]

	file:writeInt(d_sample_len)
	for j=1,d_sample_len,1 do
		file:writeFloat(d_sample[j])
	end
	file:writeInt(l_sample)
end