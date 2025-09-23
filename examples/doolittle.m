n=200;
A=rand(n,n);

L=eye(n,n);
U=zeros(n,n);

for i=1:n
    for j=i:n
        U(i,j)=A(i,j);
        for k=1:i-1
            U(i,j)=U(i,j)-L(i,k)*U(k,j);
        end
    end
    for j=i+1:n
        L(j,i)=A(j,i);
        for k=1:i-1
            L(j,i)=L(j,i)-L(j,k)*U(k,i);
        end
        L(j,i)=L(j,i)/U(i,i);
    end
end

% Verify the decomposition
norm(L*U-A)