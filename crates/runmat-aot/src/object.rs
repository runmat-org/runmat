use std::collections::BTreeSet;

use runmat_native_codegen::aot::{NativeOptimization, RelocatableNativeObject};
use runmat_types::FunctionId;

use crate::{AotError, AotResult};

mod paired;
pub use paired::{
    emit_selected_paired_native_objects, materialize_selected_paired_native_artifacts,
    PairedNativeArtifacts, PairedNativeObjects,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeObjectOptions {
    pub clock_inputs: Option<runmat_execution::host::NativeClockInputs>,
    pub local_service_inputs: Option<runmat_execution::host::NativeLocalServiceInputs>,
    pub optimization: NativeOptimization,
    pub retained_functions: Option<BTreeSet<FunctionId>>,
    pub runtime_binding_mode: runmat_native_codegen::aot::AotRuntimeBindingMode,
    pub retained_builtin_bindings: Vec<runmat_native_codegen::aot::AotBuiltinBinding>,
    pub interop: runmat_types::InteropManifest,
    pub foreign_policy: Option<runmat_execution_artifact::LogicalObject>,
    pub native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle,
    pub mex_artifacts: runmat_mex::MexArtifactBundle,
    pub java_artifacts: runmat_java::JavaArtifactBundle,
    pub python_artifacts: runmat_python::PythonArtifactBundle,
}

fn object_data_options(
    options: &NativeObjectOptions,
) -> AotResult<runmat_core::AotObjectDataOptions> {
    Ok(runmat_core::AotObjectDataOptions {
        clock_inputs: options.clock_inputs.clone(),
        local_service_inputs: options.local_service_inputs.clone(),
        runtime_binding_mode: options.runtime_binding_mode,
        builtin_bindings: options.retained_builtin_bindings.clone(),
        foreign_policy: options.foreign_policy.clone(),
        native_interfaces: options
            .native_interfaces
            .canonical_bytes()
            .map_err(|error| {
                AotError::contract("aot.compile.native_interfaces", error.to_string())
            })?,
        mex_artifacts: options
            .mex_artifacts
            .canonical_bytes()
            .map_err(|error| AotError::contract("aot.compile.mex_artifacts", error.to_string()))?,
        java_artifacts: options
            .java_artifacts
            .canonical_bytes()
            .map_err(|error| AotError::contract("aot.compile.java_artifacts", error.to_string()))?,
        python_artifacts: options
            .python_artifacts
            .canonical_bytes()
            .map_err(|error| {
                AotError::contract("aot.compile.python_artifacts", error.to_string())
            })?,
    })
}

impl Default for NativeObjectOptions {
    fn default() -> Self {
        Self {
            clock_inputs: None,
            local_service_inputs: None,
            optimization: NativeOptimization::Speed,
            retained_functions: None,
            runtime_binding_mode: runmat_native_codegen::aot::AotRuntimeBindingMode::Dynamic,
            retained_builtin_bindings: Vec::new(),
            interop: runmat_types::InteropManifest::empty(),
            foreign_policy: None,
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
    emit_object(unit, None, options)
}

/// Emit a real selected object from the same verified target-pack lease later
/// used by the selected linker. Decoded product metadata cannot call this
/// path: the selected constructor requires authenticated retained bytes.
pub fn emit_selected_native_object(
    unit: &runmat_core::ExecutableUnit,
    selected: &runmat_native_codegen::aot::SelectedNativeProduct<'_>,
    options: NativeObjectOptions,
) -> AotResult<RelocatableNativeObject> {
    if options.optimization
        != selected
            .product()
            .selected()
            .expect("verified selected constructor")
            .optimization
    {
        return Err(AotError::contract(
            "aot.compile.selected_optimization",
            "requested optimization differs from the verified selected target",
        ));
    }
    emit_object(unit, Some(selected), options)
}

fn emit_object(
    unit: &runmat_core::ExecutableUnit,
    selected: Option<&runmat_native_codegen::aot::SelectedNativeProduct<'_>>,
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
    let target = selected.map_or_else(runmat_native_codegen::NativeTarget::current, |product| {
        product.target().clone()
    });
    let assembly = input
        .lower(target)
        .map_err(|error| AotError::contract("aot.compile.lower", error.to_string()))?;
    let object_data_options = object_data_options(&options)?;
    let data = match selected {
        Some(product) => {
            input.aot_object_data_for_selected(&assembly, object_data_options, product)
        }
        None => input.aot_object_data(&assembly, object_data_options),
    }
    .map_err(|error| AotError::contract("aot.compile.data", error.to_string()))?;
    match selected {
        Some(product) => runmat_native_codegen::aot::emit_selected_relocatable_object(
            product,
            &assembly,
            data,
            options.runtime_binding_mode,
            options.retained_builtin_bindings,
        ),
        None => runmat_native_codegen::aot::emit_relocatable_object_for_runtime(
            &assembly,
            options.optimization,
            data,
            options.runtime_binding_mode,
            options.retained_builtin_bindings,
        ),
    }
    .map_err(|error| AotError::contract("aot.compile.object", error.to_string()))
}
