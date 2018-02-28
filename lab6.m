dimension=21;
x=linspace(-0.5,0.5,dimension);
y=linspace(0,0.3,dimension);
t=4;
G=0.1;
for i = 1:dimension
    for j = 1:dimension
        T(i,j)=0;
    end
end

T(1,1)=0;
T(1,dimension)=0;
T(dimension,1)=0;
T(dimension,dimension)=0;
for i=1:10000
    for m=2:(dimension-1)
        for n=2:(dimension-1)
            T(m,n)=(T(m+1,n)+T(m-1,n)+T(m,n-1)+T(m,n+1))/t+G;
        end
    end
    for j=2:(dimension-1)
        T(1,j)=(2*T(2,j)+T(1,j-1)+T(1,j+1))/t+G;
        T(dimension,j)=(2*T(dimension-1,j)+T(dimension,j-1)+T(dimension,j+1))/t+G;
    end
    for k=2:(dimension-1)
        T(k,1)=(2*T(k,2)+T(k-1,1)+T(k+1,1))/t+G;
        T(k,dimension)=(2*T(k,dimension-1)+T(k-1,dimension)+T(k+1,dimension))/t+G;
    end
end
for i = 1:dimension
    for j = 1:dimension
        T(i,j)=T(i,j)/1000;
    end
end
surf(x,y,T);
