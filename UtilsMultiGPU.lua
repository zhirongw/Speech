require 'cunn'
require 'rnn'
require 'cudnn'
require 'DPTCTC'

local default_GPU = 1
function makeDataParallel(model, nGPU, is_cudnn)
     if nGPU >= 1 then
            if is_cudnn then
                cudnn.fastest = true
                model = cudnn.convert(model, cudnn)
            end
            if nGPU > 1 then
                gpus = torch.range(1, nGPU):totable()

                dpt = nn.DPTCTC(1, true, true)
                dpt:add(model, gpus) -- now use our impl instead; nn.DataParallelTable(1)
                dpt:threads(function()
                                 require 'rnn'
                                 require 'cudnn'
                                 require 'nnx'
                                 require 'warp_ctc'
                                 require 'BNDecorator'
                             end)
                dpt.gradInput = nil
                model = dpt
            end
            model:cuda()
     end
     return model
end

local function clear(tensor)
    if tensor then
     tensor:set()
    end
end

function saveDataParallel(filename, orgModel)
    local model_type = torch.type(orgModel)
    local model
    if model_type == 'nn.DataParallelTable' or
        model_type == 'nn.DPTCTC' then
        model = orgModel:get(1):clone()
        model:clearState()
    elseif model_type == 'nn.Sequential' then
        local temp_model = nn.Sequential()
        for i, module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' or
                torch.type(module) == 'nn.DPTCTC' then
                temp_model:add(module:get(1))
            else
                temp_model:add(module)
            end
        end
        model = temp_model
    else
        assert(model_type == 'nn.gModule',
            'This saving function only works with Sequential, gModule or DataParallelTable modules.')
    end
    if torch.type(model) == 'nn.gModule' then
        for _,node in ipairs(model.forwardnodes) do
            m = node.data.module
            if m then
                if m.modules then
                    for _,inner_m in ipairs(m.modules) do
                        if torch.type(inner_m) == 'cudnn.LSTM' then
                            clear(inner_m.hiddenOutput)
                            clear(inner_m.cellOutput)
                            clear(inner_m.gradHiddenInput)
                            clear(inner_m.gradCellInput)
                            clear(inner_m.workspace)
                        else
                            inner_m.gradBias = nil
                        end
                        inner_m.gradWeight = nil
                    end
                end
                clear(m.reverse_input)
                clear(m._input)
                clear(m.reverse_gradOutput)
                clear(m._gradOutput)
            end
        end
    end
    torch.save(filename, model)
end

function loadDataParallel(filename, nGPU, is_cudnn)
    local model = torch.load(filename)
    local model_type = torch.type(model)
    if model_type == 'nn.DataParallelTable' or
        model_type == 'nn.DPTCTC' then
        return makeDataParallel(model:get(1):float(), nGPU, is_cudnn)
    elseif model_type == 'nn.Sequential' then
        for i,module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' or
                torch.type(module) == 'nn.DPTCTC' then
                model.modules[i] = makeDataParallel(module:get(1):float(), nGPU, is_cudnn)
            end
        end
        return model
    elseif model_type == 'nn.gModule' then
        model = makeDataParallel(model, nGPU, is_cudnn)
        return model
    else
        error('The loaded model is not a Sequential or DataParallelTable module.')
    end
end

