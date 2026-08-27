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
fn dense_numeric_inputs_and_outputs_keep_their_host_allocation() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("allocation_identity.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
#include <stdint.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs != 3 || nrhs != 1) mexErrMsgTxt("expected one input and three outputs");
    plhs[0] = mxCreateNumericMatrix(1, 2, mxINT32_CLASS, mxREAL);
    mxInt32 *values = mxGetInt32s(plhs[0]);
    values[0] = 17;
    values[1] = -9;
    plhs[1] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(plhs[1])[0] = (mxUint64)(uintptr_t)values;
    plhs[2] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(plhs[2])[0] = (mxUint64)(uintptr_t)mxGetData(prhs[0]);
}
"#,
    )
    .unwrap();

    let input = runmat_value::Tensor::new_integer(
        runmat_value::IntegerStorage::I32(vec![3, 4]),
        vec![1, 2],
    )
    .unwrap();
    // SAFETY: only pointer identity is observed, and the source remains alive
    // through the synchronous module invocation.
    let input_address = unsafe { input.host_buffer().foreign_data_pointer() } as usize as u64;
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(&[Value::Tensor(input)], 3, module.api_mode())
        .unwrap();

    let Value::Tensor(output) = &result.outputs[0] else {
        panic!("two-element output must remain a tensor");
    };
    // SAFETY: only pointer identity is observed while the output owns its
    // allocation.
    let output_address = unsafe { output.host_buffer().foreign_data_pointer() } as usize as u64;
    let Value::Int(created_address) = &result.outputs[1] else {
        panic!("created pointer address must remain uint64");
    };
    let Value::Int(observed_input_address) = &result.outputs[2] else {
        panic!("input pointer address must remain uint64");
    };
    assert_eq!(created_address.try_to_u64(), Some(output_address));
    assert_eq!(observed_input_address.try_to_u64(), Some(input_address));
}

#[test]
fn sparse_numeric_inputs_and_outputs_keep_compatible_host_allocations() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("sparse_allocation_identity.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
#include <stdint.h>

static mxArray *address_of(const void *pointer) {
    mxArray *value = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(value)[0] = (mxUint64)(uintptr_t)pointer;
    return value;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs != 7 || nrhs != 1 || !mxIsSparse(prhs[0])) {
        mexErrMsgTxt("expected one sparse input and seven outputs");
    }
    plhs[0] = mxCreateSparse(2, 2, 2, mxREAL);
    double *values = mxGetDoubles(plhs[0]);
    mwIndex *rows = mxGetIr(plhs[0]);
    mwIndex *columns = mxGetJc(plhs[0]);
    values[0] = 5.0;
    values[1] = -2.0;
    rows[0] = 0;
    rows[1] = 1;
    columns[0] = 0;
    columns[1] = 1;
    columns[2] = 2;
    plhs[1] = address_of(mxGetData(prhs[0]));
    plhs[2] = address_of(mxGetIr(prhs[0]));
    plhs[3] = address_of(mxGetJc(prhs[0]));
    plhs[4] = address_of(values);
    plhs[5] = address_of(rows);
    plhs[6] = address_of(columns);
}
"#,
    )
    .unwrap();

    let input =
        runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![1, 0], vec![3.0, 4.0]).unwrap();
    // SAFETY: the test observes pointer identity only while the owning sparse
    // value remains live through the synchronous invocation.
    let input_data =
        unsafe { input.numeric_host_buffer().unwrap().foreign_data_pointer() } as usize as u64;
    let input_rows = unsafe { input.row_indices.foreign_data_pointer() } as usize as u64;
    let input_columns = unsafe { input.col_ptrs.foreign_data_pointer() } as usize as u64;

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(&[Value::SparseTensor(input)], 7, module.api_mode())
        .unwrap();
    let Value::SparseTensor(output) = &result.outputs[0] else {
        panic!("first output must remain sparse");
    };
    let output_data =
        unsafe { output.numeric_host_buffer().unwrap().foreign_data_pointer() } as usize as u64;
    let output_rows = unsafe { output.row_indices.foreign_data_pointer() } as usize as u64;
    let output_columns = unsafe { output.col_ptrs.foreign_data_pointer() } as usize as u64;
    let addresses = result.outputs[1..]
        .iter()
        .map(|value| match value {
            Value::Int(value) => value.try_to_u64().unwrap(),
            _ => panic!("pointer address must remain uint64"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        addresses,
        vec![
            input_data,
            input_rows,
            input_columns,
            output_data,
            output_rows,
            output_columns
        ]
    );
}

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
fn c_gateway_compatibility_definitions_and_scalar_spellings_are_available() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("compatibility_surface.c");
    fs::write(
        &source,
        r#"
#include "mex.h"

#ifndef MATLAB_MEX_FILE
#error "MEX builds must define MATLAB_MEX_FILE"
#endif

#if MEX_INFORMATION_VERSION != 1
#error "unexpected MEX information version"
#endif

#if TARGET_API_VERSION != 700
#error "the default API pin must select the separate-complex compatibility surface"
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 1) mexErrMsgTxt("expected one output");

    bool enabled = true;
    int8_T *value = (int8_T *)malloc(sizeof(int8_T));
    if (value == NULL) mexErrMsgTxt("allocation failed");
    *value = 7;
    boolean_T flag = enabled ? 1 : 0;
    real32_T single_value = 0.5f;
    real64_T result = (real64_T)(*value + flag + abs(-3)) + single_value;
    printf("compatibility surface\n");
    plhs[0] = mxCreateDoubleScalar((real_T)result);
    free(value);
}
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module.invoke(&[], 1, module.api_mode()).unwrap();

    assert_eq!(result.outputs, vec![Value::Num(11.5)]);
    assert_eq!(result.console, "compatibility surface\n");
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
fn compatible_isolated_tier_does_not_weaken_exact_manifest_admission() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("compatible.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs == 1) plhs[0] = mxCreateDoubleScalar(42.0);
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::remove_file(&artifact.manifest).unwrap();

    assert!(matches!(
        MexModule::load(&artifact.module),
        Err(MexLoadError::ArtifactManifestRead { .. })
    ));
    let module = MexModule::load_compatible_isolated(&artifact.module).unwrap();
    let result = module.invoke(&[], 1, module.api_mode()).unwrap();
    assert_eq!(result.outputs, vec![Value::Num(42.0)]);
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

#[test]
fn completed_gateway_does_not_retain_its_invocation_host() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("host_lifetime.c");
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
    let module = MexModule::load(&artifact.module).unwrap();
    let host = Rc::new(FixtureHost::default());
    let weak = Rc::downgrade(&host);
    module
        .invoke_with_services(&[], 0, module.api_mode(), host.clone())
        .unwrap();
    drop(host);

    assert!(weak.upgrade().is_none());
}

#[test]
fn forced_shutdown_runs_at_exit_with_the_originating_host_services_alive() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("at_exit_fixture.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
static void record_exit(void) {
    mxArray *value = mxCreateDoubleScalar(99.0);
    if (mexPutVariable("base", "exit_seen", value) != 0) {
        mexErrMsgTxt("at-exit workspace callback failed");
    }
    mxDestroyArray(value);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    if (mexAtExit(record_exit) != 0) mexErrMsgTxt("could not register at-exit callback");
    mexLock();
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let host = Rc::new(FixtureHost::default());
    let module = MexModule::load(&artifact.module).unwrap();
    module
        .invoke_with_services(&[], 0, module.api_mode(), host.clone())
        .unwrap();
    assert!(module.is_locked());
    module.shutdown_with_services(host.clone()).unwrap();

    assert_eq!(
        host.workspace.lock().unwrap().get("exit_seen"),
        Some(&Value::Num(99.0))
    );
}

#[test]
fn at_exit_error_is_reported_without_crossing_the_c_abi() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("at_exit_error.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
static void fail_exit(void) {
    mexErrMsgIdAndTxt("Fixture:AtExit", "expected teardown failure");
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    mexAtExit(fail_exit);
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let host = Rc::new(FixtureHost::default());
    module
        .invoke_with_services(&[], 0, module.api_mode(), host.clone())
        .unwrap();
    let error = module
        .shutdown_with_services(host)
        .expect_err("at-exit failure must reach the host");

    assert!(matches!(
        error,
        MexLoadError::Invocation { identifier, message }
            if identifier.contains("Fixture:AtExit") && message == "expected teardown failure"
    ));
}

#[derive(Default)]
struct ReentrantHost {
    module: std::cell::RefCell<Option<std::rc::Weak<MexModule>>>,
}

impl MexHostServices for ReentrantHost {
    fn eval(&self, _command: &str) -> Result<(), MexDiagnostic> {
        unreachable!("reentrancy fixture does not evaluate source")
    }

    fn call(
        &self,
        function: &str,
        _arguments: Vec<Value>,
        _requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic> {
        assert_eq!(function, "recursive_entry");
        let module = self
            .module
            .borrow()
            .as_ref()
            .unwrap()
            .upgrade()
            .expect("fixture module is alive");
        let error = module
            .invoke(&[], 0, module.api_mode())
            .expect_err("same-module recursive entry must fail");
        Err(MexDiagnostic {
            identifier: Some("RunMat:MEX:ReentrantInvocation".into()),
            message: error.to_string(),
        })
    }

    fn get_variable(&self, _workspace: &str, _name: &str) -> Result<Option<Value>, MexDiagnostic> {
        unreachable!("reentrancy fixture does not read workspace state")
    }

    fn put_variable(
        &self,
        _workspace: &str,
        _name: &str,
        _value: Value,
    ) -> Result<(), MexDiagnostic> {
        unreachable!("reentrancy fixture does not write workspace state")
    }
}

#[test]
fn same_module_callback_reentry_fails_without_deadlocking() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("reentrant_fixture.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    if (mexCallMATLAB(0, NULL, 0, NULL, "recursive_entry") != 0) {
        mexErrMsgTxt("recursive callback rejected");
    }
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = Rc::new(MexModule::load(&artifact.module).unwrap());
    let host = Rc::new(ReentrantHost::default());
    *host.module.borrow_mut() = Some(Rc::downgrade(&module));
    let error = module
        .invoke_with_services(&[], 0, module.api_mode(), host)
        .expect_err("recursive gateway must fail");

    assert!(matches!(
        error,
        MexLoadError::Invocation { identifier, message }
            if identifier.contains("RunMat:MEX:ReentrantInvocation")
                && message.contains("recursive invocation of the same C MEX module")
    ));
}
