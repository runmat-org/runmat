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
