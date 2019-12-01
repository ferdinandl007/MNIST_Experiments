% Change the filenames if you've saved the files under different names % On some platforms, the files might be saved as
% train-images.idx3-ubyte / train-labels.idx1-ubyte
trainImages = loadMNISTImages('train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('train-labels-idx1-ubyte');


testImages = loadMNISTImages('t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');

% We are using display_network from the autoencoder code

% Show the first 100 images
%disp(trainLabels(1:10));;



% 1 for train and test 0 for lod and test
trainFlage = 1;

validationsetImage = trainImages(:,1:1000);

validationsetLabel = trainLabels(1:1000);

testImages = trainImages(:,1000:end);

testLabels = trainLabels(1000:end);

model = MLP(784, 300,10,10);

model.initWeight(1.0);

validation = [];
loss = [];
best = 10;

b_Wh = [];
b_wo = [];
hold on
if  trainFlage
    count = 0;
    step = 1;
    for Epoch = 1:4
      [w_h ,W_o] = model.getWeights();
      model.initWeight_from_mat(w_h,W_o);
      for index = 1:size(trainImages,2)
         count = (index / size(trainImages,2)) * 100;
            if (count > step)
               step = step + 1;
               loss = [loss  mean(model.getLoss())];
               error = test(validationsetImage,validationsetLabel,model);
               validation = [validation error];
               newBest = '';
               if (best < loss)
                   best = loss;
                   [b_Wh ,b_wo] = model.getWeights();
                   newBest = 'new best save model'
                   
               end 
               message = ['Epoch = ', num2str(Epoch),', Progress -> ', num2str(count),'%',', acc -> ' , num2str(error),'%', ' loss -> ',' ', num2str(mean(loss))]
               disp(newBest)
                   
            end
        
           p = zeros(10,1);
           p(trainLabels(index) + 1)  = 1.0;
           model.adapt_to_target(trainImages(:,index),p, 0.1);
      end
        count = 0;
        step = 1;
        figure(1)
        title('loss')
        xlabel('Iterations %')
        ylabel('loss')
        plot(loss,'k');
        pause(1)
        figure(2);
        title('Accuracy')
        xlabel('Iterations %')
        ylabel('Accuracy')
        plot(validation,'k');
        pause(1)
        
    end

  
    save('model_w11','b_Wh','b_wo');
    [Wh ,Wo] = model.getWeights();
    save('model_w1_l','Wh','Wo');
else 
    k = load('model_w.mat');
    model.initWeight_from_mat(k.w_h,k.W_o);
end 





error = 100 - test(testImages,testLabels,model)


function error = test(images,labels,model)
    error = 0;
    for index = 1:size(images,2)
        p = zeros(10,1);
        p(labels(index) + 1) = 1.0;
        output = model.compute_output(images(:,index));
        [output,p];
        [pm,pi] = max(p) ;
        [om,oi] = max(output);
   
        if pi == oi 
            error = error + 1;
        else 
           % display_network(images(:,index));
        end 
           
    end
    error =  error / length(images(1, 1:end)) * 100;
end


