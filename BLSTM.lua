require 'nn'
require 'nngraph'
require 'rnn'
require 'cudnn'

local layer, parent = torch.class('nn.BLSTM', 'nn.Container')

function layer:__init(inputDim, hiddenDim, isCUDNN)
    if isCUDNN then
        self.forwardModule = cudnn.LSTM(inputDim, hiddenDim, 1) 
        self.backwardModule = cudnn.LSTM(inputDim, hiddenDim, 1) 
    else
        self.forwardModule = nn.SeqLSTM(inputDim, hiddenDim) 
        self.backwardModule = nn.SeqLSTM(inputDim, hiddenDim) 
    end

    local backward = nn.Sequential()
    backward:add(nn.SeqReverseSequence(1))
    backward:add(self.backwardModule)
    backward:add(nn.SeqReverseSequence(1))

    local concat = nn.ConcatTable()
    concat:add(self.forwardModule):add(backward)
    
    local blstm = nn.Sequential()
    blstm:add(concat)
    blstm:add(nn.JoinTable(3))

    parent.__init(self)

    self.output = torch.Tensor() 
    self.gradInput = torch.Tensor() 

    self.module = blstm
    self.modules[1] = blstm
end

function layer:updateOutput(input)
    self.output = self.module:updateOutput(input)
    return self.output
end

function layer:updateGradInput(input, gradOutput)
    self.gradInput = self.module:updateGradInput(input, gradOutput)
    return self.gradInput
end

function layer:accGradParameters(input, gradOutput, scale)
    self.module:accGradParameters(input, gradOutput, scale)
end

function layer:accUpdateGradParameters(input, gradOutput, lr)
    self.module:accUpdateGradParameters(input, gradOutput, lr)
end

function layer:sharedAccUpdateGradParameters(input, gradOutput, lr)
    self.module:sharedAccUpdateGradParameters(input, gradOutput, lr)
end

function layer:__tostring__()
    if self.module.__tostring__ then
        return torch.type(self) .. ' @ ' .. self.module:__tostring__()
    else
        return torch.type(self) .. ' @ ' .. torch.type(self.module)
    end
end

BLSTM = {}
function BLSTM.createBLSTM(inputDim, hiddenDim, isCUDNN)
    
    if isCUDNN then
      return cudnn.BLSTM(inputDim, hiddenDim, 1)
    end

    local forwardmodule
    local backwardmodule
    forwardmodule = nn.SeqLSTM(inputDim, hiddenDim)
    backwardmodule = nn.SeqLSTM(inputDim, hiddenDim)
    local input = nn.Identity()()
    local forward = forwardmodule(input)
    local backward = nn.SeqReverseSequence(1)(input)
    backward = backwardmodule(backward)
    backward = nn.SeqReverseSequence(1)(backward)

    local output = nn.JoinTable(3)({forward, backward})

    return nn.gModule({input}, {output})
end

return BLSTM
