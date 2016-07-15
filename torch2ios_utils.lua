-- Kurt Jacobs
-- RandomDudes
-- 2016

torch2ios_utils = {}

function torch2ios_utils.flatten(tensor)
  local dimensions = tensor:nDimension()
  if dimensions == 1 then
    return tensor
  end

  local tensorSize = 0
  tensorSize = tensor:size(1) * tensor:size(2)
  if dimensions > 2 then
    for i=3,dimensions,1 do
      tensorSize = tensorSize * tensor:size(i)
    end
  end

  t = tensor:clone()
  t:resizeAs(torch.Tensor(tensorSize))
  return t
end