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
