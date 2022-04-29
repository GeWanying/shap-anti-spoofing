function c1_ = bigger_than_0(a)
c1_ = zeros(size(a));

for i = 1:size(a,1)
    for j = 1:size(a,2)
        if a(i,j) > 0
            c1_(i,j) = a(i,j);
        end
    end
end

