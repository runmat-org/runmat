#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::Value;
use test_helpers::execute_source;

#[test]
fn uigetdir_resolves_and_cancels_cleanly_without_dialog_provider() {
    let vars = execute_source(
        "\
        folder_img = uigetdir; \
        if ~isequal(folder_img, 0); error('expected cancellation'); end; \
        ok = 1;",
    )
    .expect("uigetdir should resolve as a builtin");

    assert!(vars.iter().any(|value| value == &Value::Num(1.0)));
    assert!(vars.iter().any(|value| value == &Value::Num(0.0)));
}
