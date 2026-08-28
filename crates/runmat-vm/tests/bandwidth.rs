#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn bandwidth_accepts_all_integer_classes_with_scalar_outputs() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "A = {constructor}([1 0 7; 0 2 0; 0 5 3]); [lower, upper] = bandwidth(A); if lower ~= 1 || upper ~= 2; error('integer bandwidth mismatch'); end; selected = bandwidth(A, 'upper'); if selected ~= 2; error('integer bandwidth selector mismatch'); end;"
        );
        execute_source(&source).expect("execute integer bandwidth script");
    }
}
