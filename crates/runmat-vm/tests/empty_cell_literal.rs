#[path = "support/mod.rs"]
mod test_helpers;

#[test]
fn empty_cell_literal_preserves_zero_by_zero_shape_in_expression_positions() {
    let code = test_helpers::compile_source(
        "C = {}; assert(iscell(C)); assert(isequal(size(C), [0 0])); \
         S = struct('id', {}); assert(isequal(size(S), [0 0])); \
         assert(isequal(size({}), [0 0]));",
    )
    .expect("empty cell literals compile");
    test_helpers::interpret(&code).expect("empty cell literals execute");
}

#[test]
fn unequal_nonempty_cell_rows_still_reject() {
    assert!(test_helpers::compile_source("C = {1; 2, 3};").is_err());
}
