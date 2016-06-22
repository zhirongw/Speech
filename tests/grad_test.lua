--[[
Courtesy of Karpathy
Unit tests for the PixelModel implementation, making sure
that nothing crashes, that we can overfit a small dataset
and that everything gradient checks.
--]]

require 'nngraph'
require 'torch'
require 'cudnn'
require 'nnx'
require '../BLSTM'

local gradcheck = require 'gradcheck'

local tests = {}
local tester = torch.Tester()

-- validates the size and dimensions of a given
-- tensor a to be size given in table sz
function tester:assertTensorSizeEq(a, sz)
  tester:asserteq(a:nDimension(), #sz)
  for i=1,#sz do
    tester:asserteq(a:size(i), sz[i])
  end
end

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

local function BRNN(feat, seqLengths, rnnModule)
    local fwdLstm = nn.MaskRNN(rnnModule:clone())({ feat, seqLengths })
    local bwdLstm = nn.ReverseMaskRNN(rnnModule:clone())({ feat, seqLengths })
--    return nn.CAddTable()({ fwdLstm, bwdLstm })
    return nn.JoinTable(2)({ fwdLstm, bwdLstm })
end

-- test just the language model alone (without the criterion)
local function gradCheckSpeech()
  torch.manualSeed(123)
  local dtype = 'torch.DoubleTensor'
  -- first build a mini model
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1,8,3,3,2,2))
  --model:add(nn.SpatialBatchNormalization(8))
  model:add(nn.ReLU(true))
  local rnnInputsize = 8 * 3
  local rnnHiddensize = 7
  model:add(nn.View(rnnInputsize, -1):setNumInputDims(3))
  model:add(nn.Transpose({2,3},{1,2}))
  
  model:add(nn.BLSTM(rnnInputsize, rnnHiddensize, false))
  model:add(nn.View(-1, 2*rnnHiddensize))
  --model = nn.BatchNormalization(rnnHiddensize)(rnn)
  model:add(nn.Linear(2*rnnHiddensize, 4))
  --model:type(dtype)

  local specs = torch.rand(2, 1, 7, 11)
  --specs[{1,1,{},{4,11}}]:fill(0)
  local sizes = torch.Tensor({1,5})
--  local labels = {{1}, {1,3,3}

  -- evaluate the analytic gradient
  local output = model:forward(specs)
  local w = torch.randn(output:size()):fill(1)
  w = w:view(5, 2, 4)
  --w[{{2,5}, 1, {}}]:fill(0)
  w = w:view(10, 4)
  local ww = w:clone()
  -- generate random weighted sum criterion
  local loss = torch.sum(torch.cmul(output, w))
  print('loss is: ' .. loss)

  local gradInput = model:backward(specs, w)

  -- create a loss function wrapper
  local function f(x)
    local output = model:forward(x)
    local loss = torch.sum(torch.cmul(output, ww))
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, specs, 1, 1e-6)

  --print(gradInput)
  --print(gradInput_num)
  --local g = gradInput:view(-1)
  --local gn = gradInput_num:view(-1)
  --for i=1,g:nElement() do
  --  local r = gradcheck.relative_error(g[i],gn[i])
  --  print(i, g[i], gn[i], r)
  --end
  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 1e-4)
end

-- test the CTC
local function gradCheckCrit()
  local dtype = 'torch.DoubleTensor'
  local seq_length = 5
  local batch_size = 3
  local feat_size = 2

  local crit = nn.CTCCriterion()
  crit:type(dtype)

  local acts = torch.rand(seq_length*batch_size, feat_size)
  local acts = torch.Tensor({{0,0,0,0,0},{1,2,3,4,5},{-5,-4,-3,-2,-1},
                        {0,0,0,0,0},{6,7,8,9,10},{-10,-9,-8,-7,-6},
                        {0,0,0,0,0},{11,12,13,14,15},{-15,-14,-13,-12,-11}})
  local labels = {{1},{3,3},{2,3}}
  local sizes = torch.Tensor({1,3,3})

  -- evaluate the analytic gradient
  local loss = crit:forward(acts, labels, sizes)
  local gradInput = crit:backward(acts, labels):double():mul(1/3)

  -- create a loss function wrapper
  local function f(x)
    local loss = crit:forward(x, labels, sizes)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, acts, 1, 1e-2)

  print(gradInput)
  print(gradInput_num)
  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
end

local function gradCheck()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.pixel_size = 3
  opt.num_mixtures = 5
  opt.recurrent_stride = 3
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 9
  opt.batch_size = 2
  opt.mult_in = true
  opt.num_neighbors = 4
  opt.border_init = 0
  local pm = nn.PixelModel4N(opt)
  local crit = nn.MSECriterion()
  pm:type(dtype)
  crit:type(dtype)

  local pixels = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size*(opt.num_neighbors+1))
  local targets = torch.rand(opt.seq_length, opt.batch_size, opt.pixel_size)
  --local borders = torch.ge(torch.rand(opt.seq_length, opt.batch_size, 1), 0.5):type(pixels:type())
  --local seq = torch.cat(pixels, borders, 3):type(dtype)

  -- evaluate the analytic gradient
  local output = pm:forward(pixels)
  local loss = crit:forward(output, targets)
  local gradOutput = crit:backward(output, targets)
  local gradInput = pm:backward(pixels, gradOutput)

  -- create a loss function wrapper
  local function f(x)
    local output = pm:forward(x)
    local loss = crit:forward(output, targets)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, pixels, 1, 1e-6)

  --print(gradInput)
  --print(gradInput_num)
  --local g = gradInput:view(-1)
  --local gn = gradInput_num:view(-1)
  --for i=1,g:nElement() do
  --local r = gradcheck.relative_error(g[i],gn[i])
  --  print(i, g[i], gn[i], r)
  --end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
end

tests.gradCheckSpeech = gradCheckSpeech
--tests.gradCheckCrit = gradCheckCrit
--tests.gradCheck = gradCheck
-- tests.overfit = overfit
--tests.sample = sample
--tests.sample_beam = sample_beam

tester:add(tests)
tester:run()
