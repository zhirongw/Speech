require 'nngraph'
require 'SeqDecorator'
require 'SplitAdd'
require 'MaskRNN'
require 'ReverseMaskRNN'
require 'UtilsMultiGPU'
--require 'rnn'

-- Chooses RNN based on if GRU or backend GPU support.
local function getRNNModule(nIn, nHidden, GRU, is_cudnn)
    if (GRU) then
        if is_cudnn then
            require 'cudnn'
            return cudnn.GRU(nIn, nHidden, 1)
        else
            require 'rnn'
        end
        return nn.GRU(nIn, nHidden)
    end
    if is_cudnn then
        require 'cudnn'
        return cudnn.LSTM(nIn, nHidden, 1)
    else
        require 'rnn'
    end
    return nn.SeqLSTM(nIn, nHidden)
end

-- Wraps rnn module into bi-directional.
local function BRNN(feat, seqLengths, rnnModule)
    local fwdLstm = nn.MaskRNN(rnnModule:clone())({ feat, seqLengths })
    local bwdLstm = nn.ReverseMaskRNN(rnnModule:clone())({ feat, seqLengths })
--    return nn.CAddTable()({ fwdLstm, bwdLstm })
    return nn.JoinTable(2)({ fwdLstm, bwdLstm })
end
-- Creates the covnet+rnn structure.
local function deepSpeech(nGPU, isCUDNN)
    local GRU = false
    local seqLengths = nn.Identity()()
    local input = nn.Identity()()
    local cnn = nn.Sequential()

    -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) conv layers.
    cnn:add(nn.SpatialConvolution(1, 32, 41, 11, 2, 2))
    cnn:add(nn.SpatialBatchNormalization(32))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialConvolution(32, 32, 21, 11, 2, 1))
    cnn:add(nn.SpatialBatchNormalization(32))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- TODO the DS2 architecture does not include this layer, but mem overhead increases.

    local rnnInputSize = 32 * 25 -- based on the above convolutions.
    local rnnHiddenSize = 400 -- size of rnn hidden layers
    local nbOfHiddenLayers = 4

    cnn:add(nn.View(rnnInputSize, -1):setNumInputDims(3)) -- batch x features x seqLength
    cnn:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features
    cnn:add(nn.View(-1, rnnInputSize)) -- (seqLength x batch) x features

    local rnn = nn.Identity()({cnn(input)})
    local rnn_module = getRNNModule(rnnInputSize, rnnHiddenSize, GRU, isCUDNN)
    rnn = BRNN(rnn, seqLengths, rnn_module)
    local rnn_module = getRNNModule(2*rnnHiddenSize, rnnHiddenSize, GRU, isCUDNN)

    for i = 1, nbOfHiddenLayers do
        rnn = nn.BatchNormalization(2*rnnHiddenSize)(rnn)
        rnn = BRNN(rnn, seqLengths, rnn_module)
    end

    rnn = nn.BatchNormalization(2*rnnHiddenSize)(rnn)
    rnn = nn.Linear(2*rnnHiddenSize, 28)(rnn)
    local model = nn.gModule({input, seqLengths}, {rnn})
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
