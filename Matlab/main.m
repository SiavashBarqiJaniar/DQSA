clc;clear;close all;

%imports

%inputs
K = 2;
N = 3;
T = 50;
M = 1;
R = 512;
channel = [];
output = [];
InputSize = 2*K + 2;

%Environment
state = [];
init = [1];
cap = [];
for j=1:K
    init = [init 0];
    cap = [cap 1];
end
init = [init cap 0];
for i=1:N
    state = [state;init];
end

%Channel
sig = ones(N , 1);
c = rayleighchan(1/10000 , 100);
c.ResetBeforeFiltering = 0;

for i = 1:K
    y = filter(c , 1);
    channel = [channel , c];
    output = [output , 20*log10(abs(y))];
    %c
end

%DQN
layers = [ ...
    sequenceInputLayer(1)
    lstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(K + 1)
    regressionLayer];

maxEpochs = 4;
miniBatchSize = 2;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0);

DQN1 = network;
DQN2 = network;
x = {[1,0,0,0,0,0] ; [1,0,0,0,0,0] ; [1,0,0,0,0,0]};
y = [[0 0 1];[0 0 0];[1 0 1]];
DQN1 = trainNetwork(x , y ,layers,options);
DQN2 = trainNetwork(x , y ,layers,options);
%predict(DQN1,x)

%RL
l = 0;
time = 1:1:100;
for i=1:R
    l =l + 1;
    Q = [];
    x = {};
    throughput = 0.0;
    for m=1:M
        data = [];
        [state , cap] = reset(state , cap);
        for t=1:T
            tq = [];
            for n=1:N
                if i<R/2
                    xx = observe(state(n,:));
                    x{end+1} = xx;
                    tq = [tq;predict(DQN1 , xx)];
                    r = uint8(rand*N);
                    
                    figure(1)
                    xlim([1 K+1]);
                    [z,a] = max(tq(n,:));
                    plot(tq(n,:))
                    hold on
                    bb = zeros(1,K+1);
                    bb(1,a) = z;
                    plot(bb,'c*');
                    %pause(0.05);
                    hold off
                    
                    
                    if mod(r,N)<K
                        a = mod(r,N)+2;
                    else
                        a = 1;
                    end
                
                else
                    xx = observe(state(n,:));
                    x{end+1} = xx;
                    tq = [tq;predict(DQN1 , xx)];
                    %plot(tq)%x(1:t),tq)
                    
                    figure(1)
                    xlim([1 K+1]);
                    [z,a] = max(tq(n,:));
                    plot(tq(n,:))
                    hold on
                    bb = zeros(1,K+1);
                    bb(1,a) = z;
                    plot(bb,'c*');
                    %pause(0.05);
                    hold off
                    
                end
                
                state(n,:) = TakeAction(state(n,:) , a);
                %data = []
                
                if n==N
                    t;
                    state = fit(state);
                    
                    %reward
                    for j=1:N
                        if state(j,1)==1
                            reward(j) = 0;
                        elseif state(j,end)==1
                            reward(j) = 1;
                        else
                            reward(j) = 0;
                        end
                    end
                    
                    NextState=[];
                    
                    %update Q
                    SetOfActions = [];
                    for j=1:N
                        %Value & Average => computing Q

                        NextState = [NextState;observe(state(j,:))];
                        Q1 = predict(DQN1 , NextState(j,:));
                        Q2 = predict(DQN2 , NextState(j,:));
                        aa = PreAction(state(j,:));
                        SetOfActions = [SetOfActions aa];
                        [z,tt] = max(Q1);
                        tq(n,aa) = reward(j) + Q2(tt);
                        Q = [Q;tq(j,:)];
                    end
                    SetOfActions = [SetOfActions 0];
                    data = [data;SetOfActions];
                    
                    %throughput
                    for j=1:N
                        if state(j,end)==1
                            throughput = throughput + (1/K);
                        end
                    end
                    
                end
                
            end
        end
        
        %Figure
        data = [data;SetOfActions];
        %figure(2)
        %pcolor(data')
        
        i
        throughput
        if throughput >= 50
            "!!!!!"
        end
    end
    
    %Train
    %x = num2cell(x);
    DQN1 = trainNetwork(x , Q ,layers,options);
    if l==5
        l = 0;
        DQN2 = trainNetwork(x , Q ,layers,options);
    end
    
end

%Functions
function y = ActionDist(q)
    alpha = 0;
    beta = 20;
    p = [];
    k = numel(q);
    pp = 0.0;
    s = 0.0;
    for j=1:k
        s = s + exp(beta*q(j));
    end
    for j=1:k
        pp = (( (1-alpha)*exp(beta*q(j)) )/s) + (alpha/k+1);
        p = [p;pp];
    end
    y = p;
end

function y = TakeAction(state , a)
    k = numel(state);
    k = k/2 - 1;
    for j=1:(k + 1)
        if j==a
            state(j) = 1;
        else
            state(j) = 0;
        end
    end
    y = state;
end

function y = observe(state)
    y = state;
end

function y = fit(state)
    [n,z] = size(state);
    [z,k] = size(state);
    k = k/2 - 1;
    cap = ones(1,k);
    for j=1:n
        for m=2:k+1
            if state(j,m)==1
                cap(m-1) = cap(m-1) - 1;
            end
        end
    end
    
    %ACK signal
    for j=1:n
        a = PreAction(state(j,:));
        if a==1
            state(j,end) = 0;
        elseif cap(a-1)<0
            state(j,end) = 0;
        else
            state(j,end) = 1;
        end
    end
    for j=1:n
        for m=1:k
            state(j,k+1+m) = cap(m);
            if cap(m)<0
                state(j,k+1+m) = 0;
            end
        end
    end
    
    y = state;
end

function y = PreAction(state)
    k = numel(state);
    k = k/2 - 1;
    for j=1:k+1
        if state(j)==1
            y = j;
        end
    end
end

function [y,w] = reset(state , cap)
    K = numel(cap);
    [N , z] = size(state);
    state = [];
    init = [1];
    cap = [];
    for j=1:K
        init = [init 0];
        cap = [cap 1];
    end
    init = [init cap 0];
    for i=1:N
        state = [state;init];
    end
    y = state;
    w = cap;
end










