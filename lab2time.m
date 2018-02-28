temp0 = zeros(11,11);
temp1 = zeros(11,11);
x = linspace(0,10,11);
y = linspace(0,10,11);
dt = 0.01;
alpha = 5;
for i = 1 : 11
    for j = 1 : 11
        temp0(i,j) = 20;
        temp1(i,j) = 20;
    end
end
for i = 1 : 11
    temp0(1,i) = 0;
    temp1(1,i) = 0;
    temp0(i,1) = 0;
    temp1(i,1) = 0;
    temp0(11,i) = 0;
    temp1(11,i) = 0;
    temp0(i,11) = 0;
    temp1(i,11) = 0;
end
mesh(x,y,temp0,'facecolor','none')
hold on
for t = 1 : 200
    for m = 2 : 10
        for n = 2 : 10
            temp1(m,n) = temp1(m,n) + (alpha * dt * ((temp1(m+1,n) + temp1(m-1,n) - (2 * temp1(m,n)))))+(alpha * dt * (temp1(m,n+1) + temp1(m,n-1) - (2 * temp1(m,n))));
        end
    end
end
mesh(x,y,temp1,'facecolor','none')