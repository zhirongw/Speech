require 'nnx'
---- inherit DataParallel table
local DPTCTC = torch.class('nn.DPTCTC','nn.DataParallelTable')

local ThreadsImplCTC = torch.class('nn.DPTCTC.Threads', 'nn.DataParallelTable.Threads')

function ThreadsImplCTC:__init(dpt, initFunc)
    self.dpt = dpt
    self.initFunc = initFunc
    self.ctc = nn.CTCCriterion():cuda()
end

function ThreadsImplCTC:applyChanges()
   if self.__threads then
      local module = self.dpt.modules[1]
      local ctc = self.ctc
      for i, gpu in ipairs(self.dpt.gpuAssignments) do
         self.__threads:addjob(i, function()
            cutorch.setDevice(gpu)
            if i == 1 then
               _G.module = module
               _G.ctc = ctc
            else
               _G.module = nil
               _G.ctc = nil
               collectgarbage()
               _G.module = module:clone()
               _G.ctc = ctc:clone()
            end
         end)
      end
      self.__threads:synchronize()
   end
end

function ThreadsImplCTC:exec(closure)
   self:setup()
   local res = {}
   for i=1,#self.dpt.gpuAssignments do
      self.__threads:addjob(i,
         function()
            return closure(_G.module, i,  _G.ctc)
         end,
         function (_res_)
            res[i] = _res_
         end)
   end
   self.__threads:synchronize()
   return res
end

local function hasFlattenedParmeters(self)
    if not self.flattenedParams then
        return false
    end
    for _, param in ipairs(self.modules[1]:parameters()) do
        if param:storage() ~= self.flattenedParams[1][1]:storage() then
            return false
        end
    end
    return true
end

-- extracts the value at idx from each entry in tbl
local function pluck(tbl, idx)
    local r = {}
    for n, val in ipairs(tbl) do
        r[n] = val[idx]
    end
    return r
end

function DPTCTC:threads(initFunc)
   require 'threads'
   self.impl:close()
   self.impl = nn.DPTCTC.Threads(self, initFunc)
   return self
end

function DPTCTC:updateOutput(input)
    if self.flattenParams and not hasFlattenedParmeters(self) then
        self:flattenParameters()
    end
    if self.needsSync then
        self:syncParameters()
    end
    local prevGpuid = cutorch.getDevice()

    -- distribute the input to GPUs
    self:_distribute(self.inputGpu, input)

    -- update output for each module
    local inputGpu = self.inputGpu
    self.outputGpu = self.impl:exec(function(m, i)
        if torch.isTensor(inputGpu[i]) and inputGpu[i]:numel() == 0 then
            return torch.CudaTensor()
        else
            return m:updateOutput(inputGpu[i])
        end
    end)
    self.output = self.outputGpu
    cutorch.setDevice(prevGpuid)

    return self.output
end

function DPTCTC:backward(input, target, size, scale)
    return self:__backward_inner('backward', input, target, size, scale)
end

function DPTCTC:updateGradInput(input, target, size)
    return self:__backward_inner('updateGradInput', input, target, size)
end

local function slice(tbl, first, last, step)
    local sliced
    if torch.type(tbl) == 'table' then
        sliced = {}
        for i = first or 1, last or #tbl, step or 1 do
            sliced[#sliced+1] = tbl[i]
        end
    else
        sliced = torch.CudaTensor()
        sliced:resize(last-first+1):copy(sizes:select(first,last))
    end
    return sliced
end

function DPTCTC:__backward_inner(method, input, target, size, scale)
    local prevGpuid = cutorch.getDevice()
    local inputGpu = self.inputGpu
    local outputGpu = self.outputGpu
    local sizeGpu = {}

    -- distribute the size to GPUs
    self:_distribute(sizeGpu, size)

    local batch_size = inputGpu[1]:size(1)
    local loss = torch.Tensor(#self.gpuAssignments)
    self.gradInputGpu = self.impl:exec(function(m, i, ctc)
        if torch.isTensor(inputGpu[i]) and inputGpu[i]:numel() == 0 then
            return torch.CudaTensor()
        else
            local targets_slice = slice(target, 1+(i-1)*batch_size, i*batch_size)
            loss[i] = ctc:forward(outputGpu[i], targets_slice, sizeGpu[i])
            local gradOutput = ctc:backward(outputGpu[i], targets_slice)
            return m[method](m, inputGpu[i], gradOutput, scale)
        end
    end)
    if method == 'backward' then
        local params = self:moduleParameters()
        -- Accumulate the gradients onto the base GPU
        if self.flattenedParams and self.usenccl and not cudaLaunchBlocking then
            if #self.gpuAssignments > 1 then
                nccl.reduce(pluck(self.flattenedParams, 2), nil, true, 1)
            end
        else
            self:_reduce(pluck(params, 2))
        end
        -- Zero out gradients on the other GPUs
        for i = 2, #self.gpuAssignments do
            cutorch.setDevice(self.gpuAssignments[i])
            for _, gradParam in ipairs(params[i][2]) do
                gradParam:zero()
            end
        end
        self.needsSync = true
    end
    cutorch.setDevice(prevGpuid)
    return loss:mean()
end

