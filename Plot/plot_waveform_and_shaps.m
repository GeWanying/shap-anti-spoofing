function plot_waveform_and_shaps(r, s_0, s_1, start, endd)

% greens = get_different_green(s_0(start:endd));
% norm_value = max(max(abs(s_0(start:endd))), max(abs(s_1(start:endd))));
% blues = get_different_green(s_0(start:endd), norm_value);
% reds = get_different_red(s_1(start:endd), norm_value);

norm_value = max(max(abs(s_0)), max(abs(s_1)));
blues = get_different_green(s_0, norm_value);
reds = get_different_red(s_1, norm_value);

non_zero_values_index = (s_0 > 0);
r_0 = r .* non_zero_values_index;
r_0(r_0 == 0) = nan;
non_zero_values_index = (s_1 > 0);
r_1 = r .* non_zero_values_index;
r_1(r_1 == 0) = nan;

x = linspace(1, size(r,2), size(r,2));

% plot(r, '-o', 'Color', [0.1 0.1 0.1])
plot(r, '-', 'LineWidth',1)
% scatter1 = scatter(x, r,'MarkerFaceColor','k','MarkerEdgeColor','k');
% scatter1.MarkerFaceAlpha = .02;
% scatter1.MarkerEdgeAlpha = .05;

xlim([start endd])
ylim([-1.2 1.2])
hold on
scatter2 = scatter(x, r_0, [], blues, 'filled');
scatter2.MarkerFaceAlpha = .7;
scatter2.MarkerEdgeAlpha = .2;
xlim([start endd])


hold on
scatter3 = scatter(x, r_1, [], reds, 'filled');
scatter3.MarkerFaceAlpha = .7;
scatter3.MarkerEdgeAlpha = .2;
xlim([start endd])

