classdef (N , K) Env
    properties
        state = [];
        init = [1];
        cap = [];
        for j=1:K
            init = [init 0];
            cap = [cap 1];
        end
        
        init = [init cap 1];
        for i=1:N
            state = [state;init];
        end
        
    end
end