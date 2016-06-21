require 'dpnn'

local layer, parent = torch.class("nn.SeqDecorator", "nn.Decorator")

function layer:__init(module)
    parent.__init(self, module)
    assert(torch.isTypeOf(module, 'nn.Module'))
end

function layer:updateOutput(input)
    self._input = input
    if self._input:nDimension() == 3 then
        self._input = self._input:view(-1, self._input:size(3))
    end
    self.output = self.module:forward(self._input)
    self.output = self.output:viewAs(input)
    return self.output
end

function layer:updateGradInput(input, gradOutput)
    self._gradOutput = gradOutput
    if self._gradOutput:nDimension() == 3 then
        self._gradOutput = self._gradOutput:view(-1, self._gradOutput:size(3))
    end
    self.gradInput = self.module:backward(self._input, self._gradOutput)
    self.gradInput = self.gradInput:viewAs(input)
    return self.gradInput
end

function layer:accGradParameters(input, gradOutput, scale)
    self.module:accGradParameters(self._input, self._gradOutput, scale)
end

function layer:accUpdateGradParameters(input, gradOutput, lr)
    self.module:accUpdateGradParameters(self._input, self._gradOutput, lr)
end

function layer:sharedAccUpdateGradParameters(input, gradOutput, lr)
    self.module:sharedAccUpdateGradParameters(self._input, self._gradOutput, lr)
end
