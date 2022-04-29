function a_ = test_diliation(a, thres)

max_p = max(max(a));
max_n = min(min(a));
a_ = a;

win_len = 3;

for i = 1+win_len:size(a,1)-win_len
    for j = 1+win_len:size(a,2)-win_len
        if a(i,j) > max_p*thres
            a_(i-win_len:i+win_len, j-win_len:j+win_len)=a_(i-win_len:i+win_len, j-win_len:j+win_len)+a(i,j);
        end
        if a(i,j) < max_n*thres
            a_(i-win_len:i+win_len, j-win_len:j+win_len)=a_(i-win_len:i+win_len, j-win_len:j+win_len)+a(i,j);
        end   
    end
end