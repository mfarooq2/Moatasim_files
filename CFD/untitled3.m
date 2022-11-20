Nx=19;
Ny=19;
iQ = nan(Nx, Ny) ;

%% Constructing pointer "vector"
id = 1:1:(Nx*Ny) ;
% id = id(randperm(Nx*Ny)) 

%% Constructing pointer "matrix"
k = 1 ;
for i = 1:1:Nx
    for j = 1:1:Ny
        iQ(i, j) = id(k) ;
        k = k + 1 ;
    end
end