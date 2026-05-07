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
