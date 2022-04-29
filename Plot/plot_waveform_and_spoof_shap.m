function plot_waveform_and_spoof_shap(r, s_0, s_1, start, endd)

% scatter_size = 18 ;
scatter_size = 35 ;
% scatter_size = 40 ;


% s_0 = s_0(start+1 :endd);
% s_1 = s_1(start+1 :endd);
% norm_value = max(abs(s_0));
norm_value = max(abs(s_0(start+1 :endd)));
% blues = get_different_green(s_0, norm_value);
noooorm =  6.5    ;
spoofs = get_different_red(s_0, norm_value/noooorm);

bonafides = get_different_green(s_1, norm_value/1.5);

non_zero_values_index = (s_0 > 0);
r_0 = r .* non_zero_values_index;
r_0(r_0 == 0) = nan;
non_zero_values_index = (s_1 > 0);
r_1 = r .* non_zero_values_index;
r_1(r_1 == 0) = nan;

x = linspace(1, size(r,2), size(r,2));

co=180;
plot(r, '-', 'Color', [co co co]/256, 'LineWidth',1.5)

xlim([start endd])
ylim([-1 1])

hold on
% x = flip(x);
% r_0 = flip(r_0);
scatter2 = scatter(x, r_0, scatter_size, spoofs, 'filled');
scatter2.MarkerFaceAlpha = .7;
scatter2.MarkerEdgeAlpha = .2;
xlim([start endd])


