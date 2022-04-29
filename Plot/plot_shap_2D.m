clear all
clf

addr = "2D-examples/";
d = dir(addr);


% change the idx to plot different files
idx =    4      ;


% load the data
d(idx).name
name = addr + d(idx).name;
file_name = d(idx).name;
id = file_name(6:end-4);

a = load(name);

% spoofed SHAP value
s_0 = squeeze(a.shap_0);

% bona fide SHAP value
s_1 = squeeze(a.shap_1);

% get the positive SHAP values for both classes
s_1 = bigger_than_0(s_1);
s_0 = bigger_than_0(s_0);

% original input spectrogram
r = squeeze(a.feature);

% set a threshold to plot only top 0.2% SHAP values
thres_2D =    0.2  ;

% If the plotted color is blue, you may need to use the commented lines here and
% in generate_shap_thres().
% resultt = generate_shap_thres(s_0, thres_2D);
resultt = -generate_shap_thres(s_0, thres_2D);
resultt = test_diliation(resultt, 0.5);
imagesc(rot90(-resultt));
colormap(gca, redblueu(10000))

% labels
xticks([1 50 100 150 200 250 300 350 400 450 500 550 600])
xticklabels({'0', '0.5', '1', '1.5', '2', '2.5','3', '3.5', '4', '4.5', '5', '5.5', '6' })
yticks([1 40 80 121])
yticklabels({'8','6','4','2' })
xlabel('Time (s)')
ylabel('Frequency (kHz)')
fsize=18;
set(gca,'FontSize', fsize, 'FontName', 'Times', 'LineWidth', 1.3)





