require 'dpnn'

local BNDecorator, parent = torch.class("nn.BNDecorator", "nn.Sequential")

function BNDecorator:__init(inputDim)
    parent.__init(self)
    self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
    self.view_out = nn.View(1, -1):setNumInputDims(2)
    self:add(self.view_in)
    self:add(nn.BatchNormalization(inputDim))
    self:add(self.view_out)
end

function BNDecorator:updateOutput(input)
    local T, N = input:size(1), input:size(2)
    self.view_in:resetSize(T * N, -1)
    self.view_out:resetSize(T, N, -1)
    return parent.updateOutput(self, input)
end

function BNDecorator:updateGradInput(input, gradOutput)
    local T, N = input:size(1), input:size(2)
    self.view_in:resetSize(T * N, -1)
    self.view_out:resetSize(T, N, -1)
    return parent.updateOutput(self, input)
end
