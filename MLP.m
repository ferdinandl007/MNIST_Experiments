classdef MLP < handle
    properties (SetAccess=private)
        inputDim
        hiddenDim
        outputDim
        
        hiddenWeights
        outputWeights
        
        hiddenBatch
        outputBatch
        batchSize
        count
        
        loss
    end
    
    
    methods
        function obj=MLP(inputD,hiddenD,outputD,bSize)
            obj.inputDim=inputD;
            obj.hiddenDim=hiddenD;
            obj.outputDim=outputD;
            obj.hiddenWeights=zeros(hiddenD,inputD+1);
             obj.outputWeights=zeros(outputD,hiddenD+1);
            obj.outputBatch = zeros(outputD,hiddenD+1);
            obj.hiddenBatch = zeros(hiddenD,inputD+1);
            obj.batchSize = bSize;
            obj.count = 0;
        end
        
        function obj=initWeight_from_mat(obj, hidden, output)
            obj.hiddenWeights = hidden;
            obj.outputWeights = output;
        end
        
        function [hiddenWeights,outputWeights]=getWeights(obj)
            hiddenWeights = obj.hiddenWeights;
            outputWeights = obj.outputWeights;
        end
        
        function obj=initWeight(obj,variance)
            obj.hiddenWeights = rand(obj.hiddenDim,obj.inputDim + 1) * (variance / 1000);
            obj.outputWeights = rand(obj.outputDim,obj.hiddenDim + 1) * variance;
        end
        
        function [hiddenNet,hidden,outputNet,output]=compute_net_activation(obj, input)
            
            input = [input; 1];
            
            hiddenNet = obj.hiddenWeights * input;
            hidden = obj.sigmoid(hiddenNet);
            
            
            input = [hidden; 1];
            
            outputNet = obj.outputWeights * input;
            output = obj.softmax(outputNet);
        end
        
        function output=compute_output(obj,input)
            [hN,h,oN,output] = obj.compute_net_activation(input);
        end
        
        function obj=adapt_to_target(obj,input,target,rate)
            [hN,h,oN,o] = obj.compute_net_activation(input);
            
             
            error = o - target;
            obj.loss = [obj.loss; error];
            dot = error .* o.* (ones(length(o), 1) - o);
            
            weights_dif_output = dot * transpose([h;1]);
            
            dh = transpose(obj.outputWeights) * dot;
            dh = dh(1:length(h)) .* (h.* (1 - h));
            %rprop
            weights_dif_hidden = dh * transpose([input; 1]);
            obj.outputBatch = obj.outputBatch + weights_dif_output;
            obj.hiddenBatch = obj.hiddenBatch + weights_dif_hidden ;
            obj.count = 1 + obj.count;
            if obj.count == obj.batchSize
                obj.count = 0;
                obj.outputWeights = obj.outputWeights - rate*(obj.outputBatch / obj.batchSize);  
                obj.hiddenWeights = obj.hiddenWeights - rate*(obj.hiddenBatch / obj.batchSize);
                obj.outputBatch = obj.outputBatch - obj.outputBatch;
                obj.hiddenBatch = obj.hiddenBatch - obj.hiddenBatch;
            end 
            
        end
        
         function obj=adapt_to_targetOnline(obj,input,target,rate)
            [hN,h,oN,o] = obj.compute_net_activation(input);
            
             
            error = o - target;
            obj.loss = [obj.loss; error];
            dot = error .* o.* (ones(length(o), 1) - o);
            
            weights_dif_output = dot * transpose([h;1]);
            
            dh = transpose(obj.outputWeights) * dot;
            dh = dh(1:length(h)) .* (h.* (1 - h));
            
            weights_dif_hidden = dh * transpose([input; 1]);
            
            obj.outputWeights = obj.outputWeights - rate*weights_dif_output;  
            obj.hiddenWeights = obj.hiddenWeights - rate*weights_dif_hidden;
         end
         
        
        function ans = getLoss(obj)
            ans = (obj.loss).^2;
            obj.loss = [];
        end
        
        function loss = CrossEntropy(yHat, y)
            if (y == 1)
                loss = -log(yHat)
            else
                loss = -log(1 - yHat)
            end
        end 
        function ans = sigmoid(obj,output)
            ans = 1 ./ (1+ exp(-output));
        end
        
        
        function soft = softmax(obj,output)
            soft = exp(output) ./ sum(exp(output));
        end
    end
end