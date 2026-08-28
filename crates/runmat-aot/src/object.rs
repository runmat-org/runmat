use std::collections::BTreeSet;

use runmat_native_codegen::aot::{NativeOptimization, RelocatableNativeObject};
use runmat_types::ProgramFunctionId;

use crate::{AotError, AotResult};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeObjectOptions {
    pub optimization: NativeOptimization,
    pub retained_functions: Option<BTreeSet<ProgramFunctionId>>,
    pub runtime_binding_mode: runmat_native_codegen::aot::AotRuntimeBindingMode,
    pub retained_builtin_bindings: Vec<runmat_native_codegen::aot::AotBuiltinBinding>,
    pub interop: runmat_types::InteropManifest,
    pub native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle,
    pub mex_artifacts: runmat_mex::MexArtifactBundle,
    pub java_artifacts: runmat_java::JavaArtifactBundle,
    pub python_artifacts: runmat_python::PythonArtifactBundle,
}

impl Default for NativeObjectOptions {
    fn default() -> Self {
        Self {
            optimization: NativeOptimization::Speed,
            retained_functions: None,
            runtime_binding_mode: runmat_native_codegen::aot::AotRuntimeBindingMode::Dynamic,
            retained_builtin_bindings: Vec::new(),
            interop: runmat_types::InteropManifest::empty(),
            native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle::empty(),
            mex_artifacts: runmat_mex::MexArtifactBundle::empty(),
            java_artifacts: runmat_java::JavaArtifactBundle::empty(),
            python_artifacts: runmat_python::PythonArtifactBundle::empty(),
        }
    }
}

pub fn emit_native_object(
    unit: &runmat_core::ExecutableUnit,
    options: NativeObjectOptions,
) -> AotResult<RelocatableNativeObject> {
    let mut input = unit
        .prepare_native_compilation_for_with_interop(None, options.interop.clone())
        .map_err(|error| {
            AotError::contract(
                "aot.compile.input",
                format!("failed to prepare canonical native input: {error}"),
            )
        })?;
    if let Some(retained) = options.retained_functions.as_ref() {
        input = input.retain_functions(retained).map_err(|error| {
            AotError::contract(
                "aot.compile.retention",
                format!("failed to apply reachability retention: {error}"),
            )
        })?;
    }
    let assembly = input
        .lower(runmat_native_codegen::NativeTarget::current())
        .map_err(|error| AotError::contract("aot.compile.lower", error.to_string()))?;
    let data = input
        .aot_object_data(
            &assembly,
            runmat_core::AotObjectDataOptions {
                runtime_binding_mode: options.runtime_binding_mode,
                builtin_bindings: options.retained_builtin_bindings.clone(),
                native_interfaces: options.native_interfaces.canonical_bytes().map_err(
                    |error| AotError::contract("aot.compile.native_interfaces", error.to_string()),
                )?,
                mex_artifacts: options.mex_artifacts.canonical_bytes().map_err(|error| {
                    AotError::contract("aot.compile.mex_artifacts", error.to_string())
                })?,
                java_artifacts: options.java_artifacts.canonical_bytes().map_err(|error| {
                    AotError::contract("aot.compile.java_artifacts", error.to_string())
                })?,
                python_artifacts: options
                    .python_artifacts
                    .canonical_bytes()
                    .map_err(|error| {
                        AotError::contract("aot.compile.python_artifacts", error.to_string())
                    })?,
            },
        )
        .map_err(|error| AotError::contract("aot.compile.data", error.to_string()))?;
    runmat_native_codegen::aot::emit_relocatable_object_for_runtime(
        &assembly,
        options.optimization,
        data,
        options.runtime_binding_mode,
        options.retained_builtin_bindings,
    )
    .map_err(|error| AotError::contract("aot.compile.object", error.to_string()))
}
