require 'nngraph'
require 'SplitAdd'
BLSTM = require 'BLSTM'
require 'UtilsMultiGPU'
require 'BNDecorator'

-- Wraps rnn module into bi-directional.
local function BLSTM(nIn, nHidden, is_cudnn)
    if is_cudnn then
        require 'cudnn'
        return cudnn.BLSTM(nIn, nHidden, 1)
    else
        require 'rnn'
        local forwardmodule
        local backwardmodule forwardmodule = nn.SeqLSTM(nIn, nHidden)
        backwardmodule = nn.SeqLSTM(nIn, nHidden)
        local input = nn.Identity()()
        local forward = forwardmodule(input)
        local backward = nn.SeqReverseSequence(1)(input)
        backward = backwardmodule(backward)
        backward = nn.SeqReverseSequence(1)(backward)
    
        local output = nn.JoinTable(3)({forward, backward})
    
        return nn.gModule({input}, {output})
    end
end


-- Creates the covnet+rnn structure.
local function deepSpeech(nGPU, isCUDNN)
    local GRU = false
    local input = nn.Identity()()
    local seqLengths = nn.Identity()()

    local model = nn.Sequential()
    -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) conv layers.
    model:add(nn.SpatialConvolution(1, 32, 41, 11, 2, 2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(32, 32, 21, 11, 2, 1))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- TODO the DS2 architecture does not include this layer, but mem overhead increases.

    local rnnInputSize = 32 * 25 -- based on the above convolutions.
    local rnnHiddenSize = 400 -- size of rnn hidden layers
    local nbOfHiddenLayers = 4

    model:add(nn.View(rnnInputSize, -1):setNumInputDims(3)) -- batch x features x seqLength
    model:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features

    model:add(BLSTM(rnnInputSize, rnnHiddenSize, isCUDNN))

    for i = 1, nbOfHiddenLayers do
        model:add(nn.BNDecorator(2*rnnHiddenSize))
        model:add(BLSTM(2*rnnHiddenSize, rnnHiddenSize, isCUDNN))
    end

    model:add(nn.View(-1, 2*rnnHiddenSize)) -- (seqLength x batch) x features
    model:add(nn.BatchNormalization(2*rnnHiddenSize))
    model:add(nn.Linear(2*rnnHiddenSize, 28))
    model = makeDataParallel(model, nGPU, isCUDNN)
    return model
end

-- Based on convolution kernel and strides.
local function calculateInputSizes(sizes)
    sizes = torch.floor((sizes - 41) / 2 + 1) -- conv1
    sizes = torch.floor((sizes - 21) / 2 + 1) -- conv2
    sizes = torch.floor((sizes - 2) / 2 + 1) -- pool1
    return sizes
end

return { deepSpeech, calculateInputSizes }
