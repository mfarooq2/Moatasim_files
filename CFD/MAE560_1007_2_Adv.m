clc, clear all, close all

%% wave propagation speed
C   = 1 ;

%% CFL
CFL = 0.5 ;

%% grid
N  = 100 ;
x  = linspace(0, 1, N+1) ;
x  = x(1:N) ;
dx = x(2) - x(1) ;

%% Delta t
dt = CFL * dx / C ;

%% initial condition
sigma = 0.01 ;
f0 = exp(- (x - 0.5).^2 / sigma);

plot(x, f0, '-', 'LineWidth', 2.5, 'Color', [0.8, 0, 0])
xlabel('$$x$$', 'Interpreter', 'LaTeX')
ylabel('$$f(x)$$', 'Interpreter', 'LaTeX')
axis([0 1 -0.2 1.2])

%% Lax-Firedichs method
f_old = f0 ;
Diff_f_old = nan(size(f0)) ;
f_bar = nan(size(f0)) ;

for it = 1:1:400    
    
    Diff_f_old(1) = (-f_old(N)   + f_old(2) ) / (2*dx) ;
    Diff_f_old(N) = (-f_old(N-1) + f_old(1) ) / (2*dx) ;
    for j = 2:1:(N-1)
        Diff_f_old(j) =  (-f_old(j-1) + f_old(j+1) ) / (2*dx) ;
    end
    
    %% Lax-Firedichs method
    f_bar(1) = (f_old(N)   + f_old(2) ) / (2) ;
    f_bar(N) = (f_old(N-1) + f_old(1) ) / (2) ;
    for j = 2:1:(N-1)
        f_bar(j) =  (f_old(j-1) + f_old(j+1) ) / (2) ;
    end
    
    f_new = f_bar - (C * dt) * Diff_f_old ;
    f_old = f_new ;
    if mod(it, 1) == 0
        plot(x, f0, 'k:', 'LineWidth', 1.5), hold on
        plot(x, f_new, '-', 'LineWidth', 2.5, 'Color', [0.8, 0, 0]), hold off
        xlabel('$$x$$', 'Interpreter', 'LaTeX')
        ylabel('$$u(x, t)$$', 'Interpreter', 'LaTeX')
        title('Central FD')
        axis([0 1 -1 2])
        pause(0.05)
    end
end