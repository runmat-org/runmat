#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use test_helpers::execute_source;

fn disable_interactive_plots_for_test() -> runmat_runtime::builtins::plotting::PlotTestLockGuard {
    let guard = runmat_runtime::builtins::plotting::lock_plot_test_context();
    runmat_runtime::builtins::plotting::reset_plot_state();
    guard
}

#[test]
fn heatmap_dot_property_assignment_routes_to_graphics_set() {
    let _guard = disable_interactive_plots_for_test();
    let input = "cdata = [45 60 32; 43 54 76; 32 94 68; 23 95 58]; \
        xvalues = {'Small','Medium','Large'}; \
        yvalues = {'Green','Red','Blue','Gray'}; \
        h = heatmap(xvalues,yvalues,cdata); \
        h.Title = 'T-Shirt Orders'; \
        h.XLabel = 'Sizes'; \
        h.YLabel = 'Colors'; \
        out = h.Title;";
    let vars = execute_source(input).expect("execute heatmap script");
    assert_eq!(vars.last(), Some(&Value::String("T-Shirt Orders".into())));
}

#[test]
fn figure_dot_property_access_routes_to_graphics_get() {
    let _guard = disable_interactive_plots_for_test();
    let input = "f = figure(); out = f.Type;";
    let vars = execute_source(input).expect("execute figure property script");
    assert_eq!(vars.last(), Some(&Value::String("figure".into())));
}

#[test]
fn groot_supports_root_handle_properties_and_dot_access() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        r = groot(); \
        if r ~= 0; error('root handle mismatch'); end; \
        if ~ishandle(r); error('root should be a handle'); end; \
        if ~isgraphics(r); error('root should be graphics'); end; \
        if ~strcmp(get(r, 'Type'), 'root'); error('root type mismatch'); end; \
        if ~strcmp(r.Type, 'root'); error('root dot type mismatch'); end; \
        if ~isempty(get(r, 'CurrentFigure')); error('unexpected current figure'); end; \
        if ~isempty(get(r, 'Parent')); error('unexpected root parent'); end; \
        f = figure(7); \
        if get(r, 'CurrentFigure') ~= f; error('current figure mismatch'); end; \
        children = get(r, 'Children'); \
        if numel(children) ~= 1 || children(1) ~= f; error('children mismatch'); end; \
        out = r.Type;";
    execute_source(input).expect("execute groot property script");
}

#[test]
fn groot_set_updates_current_figure_and_round_trips_defaults() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        r = groot(); \
        f1 = figure(11); \
        f2 = figure(12); \
        set(r, 'CurrentFigure', f1); \
        if gcf() ~= f1; error('CurrentFigure did not update'); end; \
        set(r, 'ShowHiddenHandles', 'on'); \
        if ~strcmp(get(r, 'ShowHiddenHandles'), 'on'); error('ShowHiddenHandles mismatch'); end; \
        set(r, 'Units', 'normalized'); \
        if ~strcmp(r.Units, 'normalized'); error('Units mismatch'); end; \
        set(r, 'defaultAxesTickLabelInterpreter', 'latex'); \
        if ~strcmp(get(r, 'defaultAxesTickLabelInterpreter'), 'latex'); error('default mismatch'); end; \
        props = get(r); \
        if ~strcmp(props.defaultAxesTickLabelInterpreter, 'latex'); error('default field spelling mismatch'); end; \
        out = get(r, 'defaultAxesTickLabelInterpreter');";
    execute_source(input).expect("execute groot set script");
}

#[test]
fn gobjects_preallocates_assignable_graphics_handle_arrays() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        h = gobjects(2,1); \
        if numel(h) ~= 2; error('gobjects size mismatch'); end; \
        if isgraphics(h(1)); error('placeholder should not be graphics'); end; \
        h(1) = plot(1:3, [1 4 9]); \
        tf = isgraphics(h); \
        if ~tf(1); error('assigned handle should be graphics'); end; \
        if tf(2); error('unassigned placeholder should not be graphics'); end; \
        if ~ishandle(h(1)); error('assigned handle should be a handle'); end; \
        if ishandle(h(2)); error('placeholder should not be a handle'); end; \
        h2 = gobjects([1 2]); \
        if numel(h2) ~= 2; error('size-vector form mismatch'); end; \
        out = numel(h);";
    execute_source(input).expect("execute gobjects preallocation script");
}

#[test]
fn linkaxes_propagates_limits_and_supports_off_mode() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        figure; \
        ax1 = subplot(2,1,1); \
        ax2 = subplot(2,1,2); \
        linkaxes([ax1 ax2], 'x'); \
        set(ax1, 'XLim', [0 10]); \
        xl2 = get(ax2, 'XLim'); \
        if xl2(1) ~= 0 || xl2(2) ~= 10; error('linked xlim mismatch'); end; \
        set(ax2, 'YLim', [5 9]); \
        yl1 = get(ax1, 'YLim'); \
        if yl1(1) == 5 && yl1(2) == 9; error('y should not be linked'); end; \
        linkaxes([ax1 ax2], 'xy'); \
        set(ax2, 'YLim', [2 4]); \
        yl1 = get(ax1, 'YLim'); \
        if yl1(1) ~= 2 || yl1(2) ~= 4; error('linked ylim mismatch'); end; \
        xl2_before_off = get(ax2, 'XLim'); \
        linkaxes([ax1 ax2], 'off'); \
        set(ax1, 'XLim', [20 30]); \
        xl2 = get(ax2, 'XLim'); \
        if xl2(1) ~= xl2_before_off(1) || xl2(2) ~= xl2_before_off(2); error('unlink failed'); end;";
    execute_source(input).expect("execute linkaxes script");
}

#[test]
fn tickangle_dispatches_and_round_trips_properties() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        figure; \
        ax1 = subplot(1,2,1); \
        ax2 = subplot(1,2,2); \
        xtickangle(ax1, 45); \
        if xtickangle(ax1) ~= 45; error('xtickangle scalar target failed'); end; \
        if xtickangle(ax2) ~= 0; error('xtickangle target isolation failed'); end; \
        xtickangle([ax1 ax2], -30); \
        if xtickangle(ax2) ~= -30; error('xtickangle array target failed'); end; \
        ytickangle(ax2, 25); \
        if ytickangle(ax2) ~= 25; error('ytickangle scalar target failed'); end; \
        set(get(ax2, 'YAxis'), 'TickLabelRotation', -45); \
        if ytickangle(ax2) ~= -45; error('ytickangle ruler property failed'); end; \
        if get(ax2, 'YTickLabelRotation') ~= -45; error('ytickangle axes property failed'); end;";
    execute_source(input).expect("execute tickangle script");
}

#[test]
fn figure_position_property_pair_round_trips_through_vm() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        f = figure('Position', [100 100 1000 700]); \
        p = get(f, 'Position'); \
        if p(1) ~= 100 || p(2) ~= 100 || p(3) ~= 1000 || p(4) ~= 700; \
            error('initial figure position mismatch'); \
        end; \
        set(f, 'Position', [10 20 300 400]); \
        p2 = f.Position; \
        if p2(1) ~= 10 || p2(2) ~= 20 || p2(3) ~= 300 || p2(4) ~= 400; \
            error('updated figure position mismatch'); \
        end;";
    execute_source(input).expect("execute figure Position property script");
}

#[test]
fn data_tip_text_row_dispatches_and_round_trips_properties() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        row = dataTipTextRow('Speed', 'YData', '%.2f'); \
        if ~strcmp(row.Label, 'Speed'); \
            error('label mismatch'); \
        end; \
        if ~strcmp(row.Format, '%.2f'); \
            error('format mismatch'); \
        end; \
        if ~isa(row, 'handle'); \
            error('dataTipTextRow should be handle-like'); \
        end; \
        row2 = row; \
        row.Label = 'Velocity'; \
        row.Format = '%.3f'; \
        if ~strcmp(row.Label, 'Velocity'); \
            error('updated label mismatch'); \
        end; \
        if ~strcmp(row.Format, '%.3f'); \
            error('updated format mismatch'); \
        end; \
        if ~strcmp(row2.Label, 'Velocity'); \
            error('handle alias label mismatch'); \
        end; \
        if ~strcmp(row.Value, 'YData'); \
            error('value property mismatch'); \
        end; \
        out = class(row);";
    let vars = execute_source(input).expect("execute dataTipTextRow script");
    assert!(vars.iter().any(|value| matches!(
        value,
        Value::String(class_name)
            if class_name == "matlab.graphics.datatip.DataTipTextRow"
    )));
}

#[test]
fn grid_minor_command_form_sets_minor_grid_property() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        figure; \
        plot(1:3); \
        grid minor; \
        ax = gca(); \
        if get(ax, 'MinorGrid'); \
            ok = true; \
        else; \
            error('minor grid not enabled'); \
        end;";
    execute_source(input).expect("execute grid minor command-form script");
}

#[test]
fn axis_image_command_form_enables_equal_aspect() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        figure; \
        imagesc([1 2; 3 4]); \
        axis image; \
        if get(gca, 'AxisEqual'); \
            ok = true; \
        else; \
            error('axis image did not enable equal aspect'); \
        end;";
    execute_source(input).expect("execute axis image command-form script");
}

#[test]
fn zoom_object_dispatches_and_preserves_mode_properties() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        z = zoom; \
        set(z, 'Motion', 'horizontal', 'Enable', 'on', 'ContextMenu', 123); \
        if ~strcmp(get(z, 'Enable'), 'on'); \
            error('zoom Enable mismatch'); \
        end; \
        if ~strcmp(get(z, 'Motion'), 'horizontal'); \
            error('zoom Motion mismatch'); \
        end; \
        if get(z, 'ContextMenu') ~= 123; \
            error('zoom ContextMenu mismatch'); \
        end; \
        out = class(z);";
    let vars = execute_source(input).expect("execute zoom object script");
    assert!(vars.iter().any(|value| matches!(
        value,
        Value::String(class_name)
            if class_name == "matlab.graphics.interaction.internal.zoom"
    )));
}

#[test]
fn tickformat_dispatches_and_updates_axes_properties() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        figure; \
        plot(1:3, [10 20 30]); \
        xticks([1 2 3]); \
        yticks([10 20 30]); \
        xtickformat('%.1f s'); \
        ytickformat('usd'); \
        ax = gca(); \
        xaxis = get(ax, 'XAxis'); \
        yaxis = get(ax, 'YAxis'); \
        if ~strcmp(get(xaxis, 'TickLabelFormat'), '%.1f s'); \
            error('x tick format mismatch'); \
        end; \
        if ~strcmp(get(yaxis, 'TickLabelFormat'), '$%,.2f'); \
            error('y tick format mismatch'); \
        end; \
        set(xaxis, 'TickLabelFormat', '%.2f ms'); \
        if ~strcmp(xtickformat(), '%.2f ms'); \
            error('x ruler set mismatch'); \
        end; \
        xtickformat('%.1f s'); \
        labels = xticklabels(); \
        if ~strcmp(labels{1}, '1.0 s'); \
            error('formatted x tick label mismatch'); \
        end;";
    execute_source(input).expect("execute tickformat script");
}

#[test]
fn polarplot_dispatches_and_sets_equal_axes() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        theta = linspace(0, 2*pi, 16); \
        rho = abs(sin(theta)); \
        h = polarplot(theta, rho, 'r--', 'LineWidth', 2); \
        if ~ishandle(h); \
            error('polarplot did not return a line handle'); \
        end; \
        if ~get(gca, 'AxisEqual'); \
            error('polarplot did not enable equal axes'); \
        end;";
    execute_source(input).expect("execute polarplot script");
}

#[test]
fn line_dispatches_and_round_trips_properties() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        h = line('XData', [1 2 3], 'YData', [4 5 6], 'Color', 'r', 'LineWidth', 2, 'DisplayName', 'samples'); \
        if ~ishandle(h); \
            error('line did not return a graphics handle'); \
        end; \
        if get(h, 'LineWidth') ~= 2; \
            error('line width did not round-trip'); \
        end; \
        set(h, 'XData', [7 8 9 10], 'YData', [1 1 1 1]); \
        x = get(h, 'XData'); \
        y = get(h, 'YData'); \
        if x(4) ~= 10 || y(1) ~= 1; \
            error('line data did not update'); \
        end; \
        out = get(h, 'DisplayName');";
    let vars = execute_source(input).expect("execute line property script");
    assert!(
        vars.iter()
            .any(|value| value == &Value::String("samples".into())),
        "DisplayName output missing from VM values: {vars:?}"
    );
}

#[test]
fn plotyy_dispatches_and_returns_dual_axes_outputs() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        [ax,h1,h2] = plotyy(1:3, [10 20 30], 1:3, [100 400 900], 'semilogx', 'semilogy'); \
        if numel(ax) ~= 2; \
            error('plotyy axes output mismatch'); \
        end; \
        if ~ishandle(h1) || ~ishandle(h2); \
            error('plotyy line handles invalid'); \
        end; \
        if ~strcmp(get(ax(1), 'YAxisLocation'), 'left'); \
            error('left y axis location mismatch'); \
        end; \
        if ~strcmp(get(ax(2), 'YAxisLocation'), 'right'); \
            error('right y axis location mismatch'); \
        end; \
        if ~strcmp(get(ax(1), 'XScale'), 'log'); \
            error('left x scale mismatch'); \
        end; \
        if ~strcmp(get(ax(2), 'YScale'), 'log'); \
            error('right y scale mismatch'); \
        end;";
    execute_source(input).expect("execute plotyy script");
}

#[test]
fn plotyy_preserves_subplot_parent_axes() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        figure; \
        subplot(1, 2, 2); \
        [ax,h1,h2] = plotyy(1:3, [10 20 30], 1:3, [100 400 900]); \
        if numel(ax) ~= 2; \
            error('plotyy subplot axes output mismatch'); \
        end; \
        if ~ishandle(h1) || ~ishandle(h2); \
            error('plotyy subplot line handles invalid'); \
        end; \
        if ax(1) == ax(2); \
            error('plotyy subplot axes handles were not distinct'); \
        end; \
        if ~strcmp(get(ax(2), 'YAxisLocation'), 'right'); \
            error('plotyy subplot right y axis location mismatch'); \
        end; \
        if gca() ~= ax(1); \
            error('plotyy subplot did not restore left axes as current'); \
        end;";
    execute_source(input).expect("execute subplot plotyy script");
}

#[test]
fn sphere_returns_coordinates_and_statement_form_plots() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        [X, Y, Z] = sphere(4); \
        if size(X, 1) ~= 5 || size(X, 2) ~= 5; \
            error('sphere X shape mismatch'); \
        end; \
        if size(Y, 1) ~= 5 || size(Z, 2) ~= 5; \
            error('sphere Y/Z shape mismatch'); \
        end; \
        R = sqrt(X.^2 + Y.^2 + Z.^2); \
        if max(abs(R(:) - 1)) > 1e-10; \
            error('sphere coordinates are not unit radius'); \
        end; \
        h = surf(X, Y, Z); \
        if ~ishandle(h); \
            error('surf did not accept sphere coordinate grids'); \
        end; \
        sphere(4); \
        if ~get(gca, 'AxisEqual'); \
            error('sphere statement form did not enable equal axes'); \
        end;";
    execute_source(input).expect("execute sphere script");
}

#[test]
fn bare_gca_can_set_axes_font_size() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        figure; \
        plot(1:3, [1 2 3]); \
        set(gca, 'FontSize', 10); \
        if get(gca, 'FontSize') ~= 10; \
            error('axes font size did not update'); \
        end;";
    execute_source(input).expect("execute bare gca axes font-size script");
}

#[test]
fn axes_creates_axes_and_round_trips_position_properties() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        figure; \
        ax = axes('Position', [0.2 0.3 0.4 0.5], 'Units', 'normalized'); \
        if ~ishandle(ax); \
            error('axes did not return a handle'); \
        end; \
        if ~strcmp(get(ax, 'Type'), 'axes'); \
            error('axes type mismatch'); \
        end; \
        p = get(ax, 'Position'); \
        if p(1) ~= 0.2 || p(2) ~= 0.3 || p(3) ~= 0.4 || p(4) ~= 0.5; \
            error('axes position mismatch'); \
        end; \
        if ~strcmp(ax.Units, 'normalized'); \
            error('axes units mismatch'); \
        end;";
    execute_source(input).expect("execute axes position script");
}

#[test]
fn axes_parent_property_targets_figure_and_updates_current_axes() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        f = figure(3); \
        ax = axes('Parent', f, 'Units', 'pixels', 'Position', [10 20 300 200]); \
        if get(ax, 'Parent') ~= f; \
            error('axes parent mismatch'); \
        end; \
        if gcf() ~= f; \
            error('axes parent did not update current figure'); \
        end; \
        if gca() ~= ax; \
            error('axes parent did not update current axes'); \
        end; \
        p = ax.Position; \
        if p(1) ~= 10 || p(2) ~= 20 || p(3) ~= 300 || p(4) ~= 200; \
            error('axes parent position mismatch'); \
        end;";
    execute_source(input).expect("execute axes parent script");
}

#[test]
fn axes_existing_handle_selection_preserves_properties() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        figure; \
        ax1 = axes('Units', 'normalized'); \
        ax2 = axes('Units', 'pixels', 'Position', [5 6 70 80]); \
        axes(ax1); \
        if gca() ~= ax1; \
            error('axes(ax) did not select existing axes'); \
        end; \
        axes(ax2, 'Units', 'normalized'); \
        if gca() ~= ax2; \
            error('axes(ax, props) did not select target axes'); \
        end; \
        if ~strcmp(get(ax2, 'Units'), 'normalized'); \
            error('axes(ax, props) did not apply properties'); \
        end;";
    execute_source(input).expect("execute axes selection script");
}

#[test]
fn gca_returns_active_subplot_axes_handle() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        figure; \
        ax = subplot(2, 2, 3); \
        current = gca(); \
        if current ~= ax; \
            error('gca did not return active subplot axes'); \
        end;";
    execute_source(input).expect("execute gca subplot handle script");
}

#[test]
fn gca_with_figure_handle_returns_that_figures_current_axes() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        f1 = figure(1); \
        ax1 = subplot(2, 2, 3); \
        f2 = figure(2); \
        ax2 = subplot(1, 2, 2); \
        current_f1_axes = gca(f1); \
        current_f2_axes = gca(); \
        if current_f1_axes ~= ax1; \
            error('gca(fig) did not return target figure axes'); \
        end; \
        if current_f2_axes ~= ax2; \
            error('plain gca did not keep current figure axes'); \
        end;";
    execute_source(input).expect("execute gca figure-handle script");
}

#[test]
fn gca_rejects_axes_handle_argument() {
    let _guard = disable_interactive_plots_for_test();
    let input = "\
        figure; \
        ax = subplot(2, 2, 3); \
        out = gca(ax);";
    let err = execute_source(input).expect_err("gca(ax) should reject axes handles");
    assert!(
        err.to_string().contains("expected a figure handle"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn invalid_axes_shaped_handle_member_access_reports_non_object() {
    let _guard = disable_interactive_plots_for_test();
    let input = "bad_axes_handle = 1049575; out = bad_axes_handle.Type;";
    let err = execute_source(input).expect_err("invalid axes handle should fail");
    assert!(
        err.to_string().contains("LoadMember on non-object"),
        "unexpected error: {err:?}"
    );

    let input = "bad_axes_handle = 1049575; bad_axes_handle.Title = 'bad';";
    let err = execute_source(input).expect_err("invalid axes store should fail");
    assert!(
        err.to_string().contains("StoreMember on non-object"),
        "unexpected error: {err:?}"
    );
}
