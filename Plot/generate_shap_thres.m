function temp = generate_shap_thres(c, thres)

% temp = ones(size(c));
temp = c;

topk = round(size(c,1)*size(c,2)*thres/100);
top_values = maxk(abs(c(:)), topk);
top_value = top_values(end);
max_value = top_values(1);

for i = 1:size(c,1)
    for j = 1:size(c,2)
        if c(i,j) > top_value
%             temp(i,j) = max_value - 0.2*(1-c(i,j));
            temp(i,j) = max_value - 1*(1-c(i,j));
        end
    end
end
