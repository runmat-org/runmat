#![cfg(not(target_family = "wasm"))]

use std::collections::BTreeMap;
use std::fs;
use std::rc::Rc;
use std::sync::Mutex;

use runmat_mex::{
    MexApi, MexBuild, MexDiagnostic, MexHostServices, MexLoadError, MexModule, MxApiMode,
};
use runmat_value::Value;

#[test]
fn documented_matrix_api_helpers_preserve_types_objects_and_ownership() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("matrix_api.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 4) mexErrMsgTxt("expected four outputs");

    mwSize dims[3] = {2, 3, 4};
    mxArray *indices = mxCreateUninitNumericArray(3, dims, mxUINT64_CLASS, mxREAL);
    mxUint64 *owned = (mxUint64 *)mxCalloc(24, sizeof(mxUint64));
    owned[23] = UINT64_MAX;
    mxSetUint64s(indices, owned);
    mwIndex subs[3] = {1, 2, 3};
    if (!mxIsScalar(mxCreateDoubleScalar(1.0)) ||
        mxCalcSingleSubscript(indices, 3, subs) != 23 ||
        mxGetUint64s(indices)[23] != UINT64_MAX) {
        mexErrMsgTxt("typed storage or subscript helper failed");
    }
    plhs[0] = indices;

    const char *rows[2] = {"wide", "µ"};
    plhs[1] = mxCreateCharMatrixFromStrings(2, rows);
    char *utf8 = mxArrayToUTF8String(mxCreateString("RunMat ✓"));
    if (utf8 == NULL || strcmp(utf8, "RunMat ✓") != 0) {
        mexErrMsgTxt("UTF-8 conversion failed");
    }
    mxFree(utf8);

    const char *properties[1] = {"value"};
    mxArray *object = mxCreateStructMatrix(1, 1, 1, properties);
    mxSetField(object, 0, "value", mxCreateDoubleScalar(7.0));
    if (mxSetClassName(object, "FixtureObject") != 0 ||
        !mxIsClass(object, "FixtureObject") ||
        mxGetScalar(mxGetProperty(object, 0, "value")) != 7.0) {
        mexErrMsgTxt("object property conversion failed");
    }
    mxSetProperty(object, 0, "value", mxCreateDoubleScalar(9.0));
    plhs[2] = object;

    mxArray *complex_value = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    mxGetDoubles(complex_value)[0] = 2.0;
    if (mxMakeArrayComplex(complex_value) == 0) mexErrMsgTxt("make complex failed");
#if defined(MX_HAS_INTERLEAVED_COMPLEX)
    mxGetComplexDoubles(complex_value)[0].imag = 5.0;
#else
    mxGetPi(complex_value)[0] = 5.0;
#endif
    plhs[3] = complex_value;
}
"#,
    )
    .unwrap();

    for api in [MexApi::R2017b, MexApi::R2018a] {
        let artifact = MexBuild::new(&source, directory.path())
            .api(api)
            .output_name(format!("matrix_api_{api:?}"))
            .compile()
            .unwrap();
        let module = MexModule::load(&artifact.module).unwrap();
        let result = module.invoke(&[], 4, module.api_mode()).unwrap();
        let Value::Tensor(indices) = &result.outputs[0] else {
            panic!("typed N-D result must remain a tensor");
        };
        assert_eq!(indices.shape, vec![2, 3, 4]);
        assert_eq!(
            indices.numeric_value_at(23),
            Some(runmat_value::NumericScalar::U64(u64::MAX))
        );
        let Value::CharArray(rows) = &result.outputs[1] else {
            panic!("character matrix must remain a character array");
        };
        assert_eq!(rows.shape(), &[2, 4]);
        let Value::Object(object) = &result.outputs[2] else {
            panic!("classed struct must become a RunMat object");
        };
        assert_eq!(object.class_name, "FixtureObject");
        assert_eq!(object.properties.get("value"), Some(&Value::Num(9.0)));
        assert_eq!(result.outputs[3], Value::Complex(2.0, 5.0));
    }
}

#[test]
fn api_pins_control_dimension_width_and_complex_layout() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("api_pin.c");
    fs::write(
        &source,
        r#"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 1) mexErrMsgTxt("expected one output");
    mwSize dims[2] = {1, 2};
    plhs[0] = mxCreateDoubleMatrix(dims[0], dims[1], mxREAL);
    double *values = mxGetDoubles(plhs[0]);
    values[0] = (double)sizeof(mwSize);
    values[1] = (double)mxGetN(plhs[0]);
}
"#,
    )
    .unwrap();

    for (api, expected_width, expected_mode) in [
        (
            MexApi::R2017b,
            std::mem::size_of::<usize>(),
            MxApiMode::SeparateComplex,
        ),
        (
            MexApi::R2018a,
            std::mem::size_of::<usize>(),
            MxApiMode::InterleavedComplex,
        ),
        (
            MexApi::LargeArrayDims,
            std::mem::size_of::<usize>(),
            MxApiMode::SeparateComplex,
        ),
        (
            MexApi::CompatibleArrayDims,
            std::mem::size_of::<i32>(),
            MxApiMode::SeparateComplex,
        ),
    ] {
        let output_name = format!("api_{api:?}");
        let artifact = MexBuild::new(&source, directory.path())
            .api(api)
            .output_name(output_name)
            .compile()
            .unwrap();
        let module = MexModule::load(&artifact.module).unwrap();
        assert_eq!(module.api_mode(), expected_mode);
        let result = module.invoke(&[], 1, module.api_mode()).unwrap();
        let Value::Tensor(tensor) = &result.outputs[0] else {
            panic!("API fixture must return a matrix");
        };
        assert_eq!(tensor.materialize_f64(), vec![expected_width as f64, 2.0]);
    }
}

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
    assert!(artifact.manifest.is_file());
    assert_eq!(
        runmat_mex::MexArtifactManifest::from_canonical_bytes(
            &fs::read(&artifact.manifest).unwrap()
        )
        .unwrap(),
        artifact.artifact
    );
    artifact
        .artifact
        .validate_module(&fs::read(&artifact.module).unwrap())
        .unwrap();
    let rebuilt = MexBuild::new(&source, directory.path()).compile().unwrap();
    assert_eq!(rebuilt.manifest, artifact.manifest);
    assert_eq!(
        runmat_mex::MexArtifactManifest::from_canonical_bytes(
            &fs::read(&rebuilt.manifest).unwrap()
        )
        .unwrap(),
        rebuilt.artifact
    );
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(
            &[Value::Int(runmat_value::IntValue::U64(
                9_007_199_254_740_993,
            ))],
            1,
            module.api_mode(),
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
fn loader_rejects_a_module_that_no_longer_matches_its_artifact_identity() {
    use std::io::Write as _;

    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("tamper.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::OpenOptions::new()
        .append(true)
        .open(&artifact.module)
        .unwrap()
        .write_all(b"tampered")
        .unwrap();

    let error = match MexModule::load(&artifact.module) {
        Ok(_) => panic!("loader admitted a module that did not match its manifest"),
        Err(error) => error,
    };
    assert!(matches!(error, MexLoadError::ArtifactManifest { .. }));
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
    let error = module.invoke(&[], 0, module.api_mode()).unwrap_err();
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
    let result = module.invoke(&[], 2, module.api_mode()).unwrap();
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
    for api in [MexApi::R2017b, MexApi::CompatibleArrayDims] {
        let artifact = MexBuild::new(&source, directory.path())
            .api(api)
            .output_name(format!("sparse_{api:?}"))
            .compile()
            .unwrap();
        let module = MexModule::load(&artifact.module).unwrap();
        let result = module.invoke(&[], 1, module.api_mode()).unwrap();
        let Value::SparseTensor(value) = &result.outputs[0] else {
            panic!("output must be sparse");
        };
        assert_eq!(value.col_ptrs, vec![0, 1, 2]);
        assert_eq!(value.row_indices, vec![1, 0]);
        assert_eq!(value.nnz(), 2);
    }
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
    let first = module.invoke(&[], 1, module.api_mode()).unwrap();
    let second = module.invoke(&[], 1, module.api_mode()).unwrap();
    assert_eq!(first.outputs, vec![Value::Num(1.0)]);
    assert_eq!(second.outputs, vec![Value::Num(2.0)]);
    assert!(module.is_locked());
    assert!(!module.clear().unwrap());
    module
        .invoke(&[Value::Num(0.0)], 0, module.api_mode())
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
            module.api_mode(),
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
