function c1_ = bigger_than_percentage(A, percentage)

B = reshape(A, [], 1);

C = sort(abs(B));

num_ele = size(B,1);


num_ele2 = round(num_ele * percentage) - 1 ;

max_value = C(num_ele2);

c1_ = zeros(size(A));

for i = 1:size(A,1)
    for j = 1:size(A,2)
        if A(i,j) > max_value
            c1_(i,j) = A(i,j);
        end
        if A(i,j) < -max_value
            c1_(i,j) = A(i,j);
        end
    end
end