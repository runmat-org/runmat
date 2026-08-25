#![cfg(not(target_family = "wasm"))]

use std::collections::BTreeMap;
use std::fs;
use std::rc::Rc;
use std::sync::Mutex;

use runmat_mex::{MexBuild, MexDiagnostic, MexHostServices, MexLoadError, MexModule, MxApiMode};
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

#[test]
fn cell_and_struct_ownership_crosses_the_gateway_once() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("containers.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 2) mexErrMsgTxt("expected two outputs");
    plhs[0] = mxCreateCellMatrix(1, 1);
    mxSetCell(plhs[0], 0, mxCreateDoubleScalar(7.0));
    const char *fields[] = {"value"};
    plhs[1] = mxCreateStructMatrix(1, 1, 1, fields);
    mxSetField(plhs[1], 0, "value", mxCreateLogicalScalar(1));
    if (mxGetScalar(mxGetCell(plhs[0], 0)) != 7.0 ||
        !mxIsLogicalScalarTrue(mxGetFieldByNumber(plhs[1], 0, 0))) {
        mexErrMsgTxt("nested array lookup failed");
    }
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(&[], 2, MxApiMode::InterleavedComplex)
        .unwrap();
    let Value::Cell(cell) = &result.outputs[0] else {
        panic!("first output must be a cell array");
    };
    assert_eq!(cell.data, vec![Value::Num(7.0)]);
    let Value::Struct(structure) = &result.outputs[1] else {
        panic!("second output must be a struct");
    };
    assert_eq!(structure.fields.get("value"), Some(&Value::Bool(true)));
}

#[test]
fn sparse_capacity_uses_column_pointers_as_the_actual_nonzero_boundary() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("sparse_fixture.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 1) mexErrMsgTxt("expected one output");
    plhs[0] = mxCreateSparse(3, 2, 4, mxREAL);
    mwIndex *ir = mxGetIr(plhs[0]);
    mwIndex *jc = mxGetJc(plhs[0]);
    double *values = mxGetDoubles(plhs[0]);
    ir[0] = 1; values[0] = 4.0;
    ir[1] = 0; values[1] = 8.0;
    jc[0] = 0; jc[1] = 1; jc[2] = 2;
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(&[], 1, MxApiMode::InterleavedComplex)
        .unwrap();
    let Value::SparseTensor(value) = &result.outputs[0] else {
        panic!("output must be sparse");
    };
    assert_eq!(value.col_ptrs, vec![0, 1, 2]);
    assert_eq!(value.row_indices, vec![1, 0]);
    assert_eq!(value.nnz(), 2);
}

#[test]
fn persistent_arrays_and_locks_are_owned_by_the_loaded_module() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("persistent_fixture.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
static mxArray *counter = NULL;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)prhs;
    if (counter == NULL) {
        counter = mxCreateDoubleScalar(0.0);
        mexMakeArrayPersistent(counter);
        mexLock();
    }
    mxGetDoubles(counter)[0] += 1.0;
    if (nlhs == 1) plhs[0] = mxDuplicateArray(counter);
    if (nrhs == 1) mexUnlock();
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let first = module
        .invoke(&[], 1, MxApiMode::InterleavedComplex)
        .unwrap();
    let second = module
        .invoke(&[], 1, MxApiMode::InterleavedComplex)
        .unwrap();
    assert_eq!(first.outputs, vec![Value::Num(1.0)]);
    assert_eq!(second.outputs, vec![Value::Num(2.0)]);
    assert!(module.is_locked());
    assert!(!module.clear().unwrap());
    module
        .invoke(&[Value::Num(0.0)], 0, MxApiMode::InterleavedComplex)
        .unwrap();
    assert!(module.clear().unwrap());
}

#[derive(Default)]
struct FixtureHost {
    workspace: Mutex<BTreeMap<String, Value>>,
}

impl MexHostServices for FixtureHost {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        if command == "accepted" {
            Ok(())
        } else {
            Err(MexDiagnostic {
                identifier: Some("Fixture:Eval".into()),
                message: command.into(),
            })
        }
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic> {
        if function == "plus_one" && requested_outputs == 1 {
            let Value::Num(value) = arguments[0] else {
                panic!("fixture expected numeric scalar");
            };
            return Ok(vec![Value::Num(value + 1.0)]);
        }
        Err(MexDiagnostic {
            identifier: Some("Fixture:Call".into()),
            message: function.into(),
        })
    }

    fn get_variable(&self, _workspace: &str, name: &str) -> Result<Option<Value>, MexDiagnostic> {
        Ok(self.workspace.lock().unwrap().get(name).cloned())
    }

    fn put_variable(
        &self,
        _workspace: &str,
        name: &str,
        value: Value,
    ) -> Result<(), MexDiagnostic> {
        self.workspace.lock().unwrap().insert(name.into(), value);
        Ok(())
    }
}

#[test]
fn callbacks_and_workspace_access_route_through_the_explicit_host_port() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("callback_fixture.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs != 2 || nrhs != 1) mexErrMsgTxt("expected two outputs and one input");
    mxArray *arguments[] = {(mxArray *)prhs[0]};
    mxArray *called[] = {NULL};
    mexCallMATLAB(1, called, 1, arguments, "plus_one");
    mexPutVariable("base", "saved", called[0]);
    plhs[0] = mexGetVariable("base", "saved");
    mexEvalString("accepted");
    plhs[1] = mexEvalStringWithTrap("trapped");
    if (plhs[1] == NULL || !mxIsStruct(plhs[1])) {
        mexErrMsgTxt("trap did not return an exception value");
    }
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke_with_services(
            &[Value::Num(8.0)],
            2,
            MxApiMode::InterleavedComplex,
            Rc::new(FixtureHost::default()),
        )
        .unwrap();
    assert_eq!(result.outputs[0], Value::Num(9.0));
    let Value::Struct(exception) = &result.outputs[1] else {
        panic!("trap output must be an exception structure");
    };
    assert_eq!(
        exception.fields.get("identifier"),
        Some(&Value::CharArray(runmat_value::CharArray::new_row(
            "Fixture:Eval"
        )))
    );
}
