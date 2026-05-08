#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use runmat_parser::parse;
use test_helpers::{execute, lower};

#[test]
fn heatmap_dot_property_assignment_routes_to_graphics_set() {
    unsafe {
        std::env::set_var("RUNMAT_DISABLE_INTERACTIVE_PLOTS", "1");
    }
    let input = "cdata = [45 60 32; 43 54 76; 32 94 68; 23 95 58]; \
        xvalues = {'Small','Medium','Large'}; \
        yvalues = {'Green','Red','Blue','Gray'}; \
        h = heatmap(xvalues,yvalues,cdata); \
        h.Title = 'T-Shirt Orders'; \
        h.XLabel = 'Sizes'; \
        h.YLabel = 'Colors'; \
        out = h.Title;";
    let ast = parse(input).expect("parse heatmap script");
    let hir = lower(&ast).expect("lower heatmap script");
    let vars = execute(&hir).expect("execute heatmap script");
    assert_eq!(vars.last(), Some(&Value::String("T-Shirt Orders".into())));
}

#[test]
fn figure_dot_property_access_routes_to_graphics_get() {
    unsafe {
        std::env::set_var("RUNMAT_DISABLE_INTERACTIVE_PLOTS", "1");
    }
    let input = "f = figure(); out = f.Type;";
    let ast = parse(input).expect("parse figure property script");
    let hir = lower(&ast).expect("lower figure property script");
    let vars = execute(&hir).expect("execute figure property script");
    assert_eq!(vars.last(), Some(&Value::String("figure".into())));
}

#[test]
fn invalid_axes_shaped_handle_member_access_reports_non_object() {
    unsafe {
        std::env::set_var("RUNMAT_DISABLE_INTERACTIVE_PLOTS", "1");
    }
    let input = "bad_axes_handle = 1049575; out = bad_axes_handle.Type;";
    let ast = parse(input).expect("parse invalid axes handle script");
    let hir = lower(&ast).expect("lower invalid axes handle script");
    let err = execute(&hir).expect_err("invalid axes handle should fail");
    assert!(
        err.to_string().contains("LoadMember on non-object"),
        "unexpected error: {err:?}"
    );

    let input = "bad_axes_handle = 1049575; bad_axes_handle.Title = 'bad';";
    let ast = parse(input).expect("parse invalid axes store script");
    let hir = lower(&ast).expect("lower invalid axes store script");
    let err = execute(&hir).expect_err("invalid axes store should fail");
    assert!(
        err.to_string().contains("StoreMember on non-object"),
        "unexpected error: {err:?}"
    );
}
