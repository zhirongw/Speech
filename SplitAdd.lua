require 'nn'
require 'nngraph'

local layer, parent = torch.class('nn.SplitAdd', 'nn.Module')

function layer:__init()
    parent:__init()
end

function layer:updateOutput(input)
    assert(input:nDimension() == 3, 'We only assume the input to have 3 dimensions')
    local feat_length = input:size(3)
    local left = input:narrow(3, 1, feat_length/2)
    local right = input:narrow(3, feat_length/2+1, feat_length/2)
    self.output = torch.add(left, right)
    return self.output
end

function layer:updateGradInput(input, gradOutput)
    self.gradInput = torch.repeatTensor(gradOutput, 1, 1, 2)
    return self.gradInput
end
