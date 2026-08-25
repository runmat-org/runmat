#![cfg(not(target_family = "wasm"))]

use std::fs;

use runmat_mex::{MexBuild, MexLoadError, MexModule, MxApiMode};
use runmat_value::Value;

#[test]
fn independently_compiled_gateway_loads_and_preserves_typed_input() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("add_one.c");
    fs::write(
        &source,
        r#"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1 || nlhs != 1 || !mxIsUint64(prhs[0])) {
        mexErrMsgIdAndTxt("RunMat:Fixture:Arguments", "expected one uint64 input and output");
    }
    mexPrintf("typed fixture\n");
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(plhs[0])[0] = mxGetUint64s(prhs[0])[0] + 1;
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(
            &[Value::Int(runmat_value::IntValue::U64(
                9_007_199_254_740_993,
            ))],
            1,
            MxApiMode::InterleavedComplex,
        )
        .unwrap();
    assert_eq!(
        result.outputs,
        vec![Value::Int(runmat_value::IntValue::U64(
            9_007_199_254_740_994
        ))]
    );
    assert_eq!(result.console, "typed fixture\n");
}

#[test]
fn mex_error_stops_the_gateway_without_unwinding_through_rust() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("failure.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    mexErrMsgIdAndTxt("Fixture:Expected", "failure %d", 42);
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let error = module
        .invoke(&[], 0, MxApiMode::InterleavedComplex)
        .unwrap_err();
    assert!(matches!(
        error,
        MexLoadError::Invocation { identifier, message }
            if identifier.contains("Fixture:Expected") && message == "failure 42"
    ));
}
