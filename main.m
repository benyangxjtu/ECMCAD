close all; clear all; clc
warning off;
addpath(genpath('ClusteringMeasure'));
addpath(genpath('utils'));
addpath(genpath('./datasets/'));
load('mnist4')


numview = length(X);
for i = 1:numview
    X{i} = mapstd(double(X{i}));
    [N,m] = size(X{i});
end
tic;
NC=length(unique(Y));%number of category
M=14; %numebr of anchor 
RESULT=[];
[B,runtime1] = GenerateB(X, M);
for kkkk=1:1
    lambdalist=[1e3];
    gammalist=[1.2];
    for zzz=1:length(gammalist)
        for zzzz=1:length(lambdalist)
         [Y_pred, Obj, J_maxtrix] = ECMCAD(B,NC,gammalist(zzz),lambdalist(zzzz));
         [~ , label] = max(Y_pred, [], 2);
        t=toc;
        result=ClusteringMeasure(Y,label);
        result=[gammalist(zzz),lambdalist(zzzz),result]
        RESULT=[RESULT;result];
        end
    end
end
record=[mean(RESULT(:,1)),std(RESULT(:,1));
 mean(RESULT(:,2)),std(RESULT(:,2));
 mean(RESULT(:,3)),std(RESULT(:,3));
 mean(RESULT(:,4)),std(RESULT(:,4));
 mean(RESULT(:,5)),std(RESULT(:,5));
 mean(RESULT(:,6)),std(RESULT(:,6));
 mean(RESULT(:,7)),std(RESULT(:,7));]
record=record'