require 'dpnn'

local layer, parent = torch.class("nn.LSTMDecorator", "nn.Decorator")

function layer:__init(module)
    parent.__init(self, module)
    assert(torch.isTypeOf(module, 'nn.Module'))
end

function layer:updateOutput(input)
    self._input = input[1]:view(-1, input[2]:size(1), input[1]:size(2))
    self.output = self.module:updateOutput(self._input)
    self.output = self.output:view(self._input:size(1) * self._input:size(2), -1)
    return self.output
end

function layer:updateGradInput(input, gradOutput)
    self._gradOutput = gradOutput:view(self._input:size(1), input[2]:size(1), -1)
    self.gradInput = self.module:updateGradInput(self._input, self._gradOutput)
    self.gradInput = self.gradInput:viewAs(input[1])
    return { self.gradInput, nil }
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
