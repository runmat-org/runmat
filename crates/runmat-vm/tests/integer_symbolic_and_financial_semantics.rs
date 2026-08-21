#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_limit_accepts_exact_integer_points_and_gates_numeric_expressions() {
    {
        let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        execute_source("syms x; y=limit(x,x,uint64(2));").expect("documented integer limit point");
        let error = execute_source("y=limit(uint8(1));")
            .expect_err("numeric expression is a RunMat extension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:LimitNumericExpressionExtension")
        );
    }
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let error = execute_source("y=limit(uint8(1));")
        .expect_err("constant expression has no variable to infer");
    assert_eq!(error.identifier(), Some("RunMat:limit:InvalidVariable"));
}

#[test]
fn compiled_macd_integer_matrix_extension_is_gated_and_returns_double() {
    {
        let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        let error = execute_source("y=macd(uint16([2 1 1 1;3 2 2 2;4 3 3 4]));")
            .expect_err("integer price matrix is a RunMat extension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:MacdNondoubleMatrixExtension")
        );
    }
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source("y=macd(uint16([2 1 1 1;3 2 2 2;4 3 3 4])); if ~isa(y,'double') || ~isequal(size(y),[3 1]); error('macd integer result'); end;")
        .expect("RunMat integer price matrix extension");
}
