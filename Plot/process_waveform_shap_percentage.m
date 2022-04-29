function [bonafide_percentage, spoof_percentage] = process_waveform_shap_percentage(name, top)

a = load(name);
s_0 = squeeze(a.shap_0);
s_1 = squeeze(a.shap_1);
r = squeeze(a.feature);
fs = 16000;
windowDuration = 0.03;
numWindowSamples = round(windowDuration*fs);
win = hamming(numWindowSamples,'periodic');
percentOverlap = 80;
overlap = round(numWindowSamples*percentOverlap/100);
speechIndices = detectSpeech(r',fs,"Window",win,"OverlapLength",overlap);


s_0 = bigger_than_0(s_0);
s_1 = bigger_than_0(s_1);

percentage =      top  ;
s_0 = bigger_than_percentage(s_0, percentage);
s_1 = bigger_than_percentage(s_1, percentage);
spoof_all = nnz(s_0);
bonafide_all = nnz(s_1);

bonafide_index = find(s_1>0);
spoof_index = find(s_0>0);

spoof_count = 0;
for i = 1:size(spoof_index, 2)
    flag = in_or_not(spoof_index(i), speechIndices);
    spoof_count = spoof_count+flag;
end
spoof_percentage = spoof_count / spoof_all;

bonafide_count = 0;
for i = 1:size(bonafide_index, 2)
    flag = in_or_not(bonafide_index(i), speechIndices);
    bonafide_count = bonafide_count+flag;
end
bonafide_percentage = bonafide_count / bonafide_all;