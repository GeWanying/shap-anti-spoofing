function IN = in_or_not(idx, speechIndices)
IN = 0 ;
for i =1:size(speechIndices, 1)
    if idx >= (speechIndices(i,1)) && idx <= (speechIndices(i,2))
        IN = 1;
    end
end
