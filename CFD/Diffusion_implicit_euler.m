clc, clear all, close all

%% diffusivity
a  = 0.005 ;

%% Boundary conditions
DBC_L = 1 ;
DBC_R = 2 ;

%% Delta t
dt = 0.02 ;


%% grid
N  = 49 ;
x  = linspace(0, 1, N+2)' ;
x  = x(2:N+1) ;
dx = x(2) - x(1) ;

%% initial condition
f0 = zeros(size(x)) ;

plot(x, f0, '-', 'LineWidth', 2.5, 'Color', [0.8, 0, 0])
xlabel('$$x$$', 'Interpreter', 'LaTeX')
ylabel('$$f(x)$$', 'Interpreter', 'LaTeX')
text(0.1, 2, sprintf('Time = %2.4f', 0))
axis([0 1 -0.3 2.5])


%% Build LHS matrix (matrix A)
p = a * dt / (dx^2) ;
A =   diag(-p * ones(N-1, 1), -1) ...
    + diag(2 * p * ones(N, 1), 0) ...
    + diag(-p * ones(N-1, 1), +1) ;
A = A + eye(size(A)) ;


%% Build BC vector
u_bc = zeros(size(x)) ;
u_bc(1) = a * dt / (dx^2) * DBC_L ;
u_bc(N) = a * dt / (dx^2) * DBC_R ;


%% Time stepping
f_old = f0 ;

for it = 1:1:10000  
    
    %% implicit Euler
    b = u_bc + f_old ;
    f_new = Func_GaussElimination(A, b) ;
    f_old = f_new ;
    
    if mod(it, 20) == 0
        % plot initial condition
        plot(x, f0, 'r--', 'LineWidth', 1.5, 'Color', [0.8, 0, 0]), hold on
        
        % append BC for plotting
        f_plot = [DBC_L; f_new; DBC_R] ;
        
        plot([0; x; 1], f_plot, '-', 'LineWidth', 2.5, 'Color', [0.8, 0, 0]), hold off
        xlabel('$$x$$', 'Interpreter', 'LaTeX')
        ylabel('$$f(x, t)$$', 'Interpreter', 'LaTeX')
        text(0.1, 2, sprintf('Time = %2.4f', it * dt), 'FontSize', 18)

        axis([0 1 -0.3 2.5])
        pause(0.0001)
    end
end


function return_x = Func_GaussElimination( A, b )
% Simple Gaussian Elimination program

% Find the number of linear equations
n = size(A, 1) ;

% Create augmented matrix
G = [A b];

fac = nan(n, 1) ;

% Start row elimination
for i = 1:n-1
    
    % Pivoting: swith the positions of equation to avoid zero element on diagonal
    if G(i,i) == 0
        for k = i+1:n
            if G(k,i) ~= 0                
                tmp = G(k,:);
                G(k,:) = G(i,:);
                G(i,:) = tmp;
                
                break;
            end
        end
    end
    
    fac(i+1:n) = -G(i+1:n,i)/G(i,i) ;
    G(i+1:n,:) = G(i+1:n,:) + fac(i+1:n)*G(i,:) ;
end

% Backward substitution
return_x(n,1) = G(n,n+1)/G(n,n);
for i = n-1:-1:1
    return_x(i,1) = (G(i,n+1)-G(i,i+1:n)*return_x(i+1:n))/G(i,i);
end

end
