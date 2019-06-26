%run plethora of tests
XY = xlsread("3-2train.xlsx");
train = XY;
test = XY;
pre = xlsread("3-2pre.xlsx");

train_X = train(:,1:21);
train_Y = train(:,26);
test_X = test(:,1:21);
test_Y = test(:,26);
pre_X = pre;

clear extra_options
    extra_options.importance = 1; %(0 = (Default) Don't, 1=calculate)
   
%训练模型
model=regRF_train(train_X,train_Y,500,4, extra_options);
%测试模型
modeltest_Y = regRF_predict(test_X,model);
%模型精度：RMSE
RMSE =  sqrt(sum((modeltest_Y-test_Y).^2)/632);
fprintf('\n MSE rate %f\n',   sqrt(sum((modeltest_Y-test_Y).^2)/632));
%模型预测
pre_Y = regRF_predict(pre_X,model);

%写入excel
xlswrite("3-2_Si-result.xlsx",test_Y,1);
xlswrite("3-2_Si-result.xlsx",modeltest_Y,2);
xlswrite("3-2_Si-result.xlsx",pre_Y,3);
xlswrite("3-2_Si-result.xlsx",RMSE,4);

%模型精度评价
 figure('Name','Importance Plots')
    subplot(3,1,1);
    bar(model.importance(:,end-1));xlabel('feature');ylabel('magnitude');
    title('Mean decrease in Accuracy');
    
    subplot(3,1,2);
    bar(model.importance(:,end));xlabel('feature');ylabel('magnitude');
    title('Mean decrease in Gini index');
    
    
    %importanceSD = The ?standard errors? of the permutation-based importance measure. For classification,
    %           a D by nclass + 1 matrix corresponding to the first nclass + 1
    %           columns of the importance matrix. For regression, a length p vector.
    model.importanceSD
    subplot(3,1,3);
    bar(model.importanceSD);xlabel('feature');ylabel('magnitude');
    title('Std. errors of importance measure');

    % OOB error
    title('Std. errors of importance measure');
     figure('Name','OOB error rate');
    plot(model.mse); title('OOB MSE error rate');  xlabel('iteration (# trees)'); ylabel('OOB error rate');
    