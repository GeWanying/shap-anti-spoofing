function c = get_different_red(input, norm_value)
input_norm = input / norm_value;
total_number = max(size(input,1), size(input,2));

c = zeros(total_number,3);
for i=1:total_number
%     c(i,:) = [abs(input_norm(i)) 0 0];
    c(i,:) = [1 1-abs(input_norm(i)) 1-abs(input_norm(i))];
end