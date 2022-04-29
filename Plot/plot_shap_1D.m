clear all
clf

addr = "1D-examples/";

d = dir(addr);

% change the idx to plot different files
idx =  8  ;

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

% original waveform
r = squeeze(a.feature);
% sound(r, 16000)

% get the positive SHAP values for both classes
s_0 = bigger_than_0(s_0);
s_1 = bigger_than_0(s_1);

% set a threshold to plot only top 0.2% SHAP values
percentage =      0.998 ;
s_0 = bigger_than_percentage(s_0, percentage);
s_1 = bigger_than_percentage(s_1, percentage);

% start point in samples
s =    0     ;
% end point in samples, default is audio length
diff =      length(s_1)      ;

% plot gray scale waveform
% co =    180  ;
% plot(r, '-', 'LineWidth', 1.5, 'Color',[co co co]/256)
% xlim([1 length(s_0)])

% plot SHAP value on top of the waveform
plot_waveform_and_spoof_shap(r, s_0, s_1, s, s + diff);

% labels
xticks([1 16000*0.5 16000*1 16000*1.5 16000*2 16000*2.5 16000*3 16000*3.5 16000*4 16000*4.5 16000*5 16000*5.5 16000*6])
xticklabels({'0' '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0' })
xlabel('Time (s)')
ylabel('Amplitude')
set(gca,'TickDir','out');
set(gca,'box','off')
fsize=18;
set(gca,'FontSize', fsize, 'FontName', 'Times', 'LineWidth', 1.3)