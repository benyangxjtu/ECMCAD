function [B,runtime1] = GenerateB(fea, num_anchor)
tic
cfea = [];
for t=1:length(fea)
    cfea = [fea{t},cfea];
end
% fea{t+1} = cfea;
num_view = length(fea);
B = cell(1, num_view);
L = cell(1, num_view);
S = cell(1, num_view);
D = cell(1, num_view);
[~, cluster_centers] = litekmeans(cfea, num_anchor);
temp1=[];
for i=1:num_view
    temp0=size(fea{i},2);
    temp1=[temp1,temp0];
end
cluster_centers_cell=mat2cell(cluster_centers,num_anchor,temp1);
for t=1:num_view
    B{t} = ConstructA_NP(fea{t}', cluster_centers_cell{t}');
end
runtime1 = toc;
end

