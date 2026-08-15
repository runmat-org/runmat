use std::collections::BTreeSet;

use runmat_native_codegen::aot::{NativeOptimization, RelocatableNativeObject};
use runmat_types::ProgramFunctionId;

use crate::{AotError, AotResult};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeObjectOptions {
    pub optimization: NativeOptimization,
    pub retained_functions: Option<BTreeSet<ProgramFunctionId>>,
}

impl Default for NativeObjectOptions {
    fn default() -> Self {
        Self {
            optimization: NativeOptimization::Speed,
            retained_functions: None,
        }
    }
}

pub fn emit_native_object(
    unit: &runmat_core::ExecutableUnit,
    options: NativeObjectOptions,
) -> AotResult<RelocatableNativeObject> {
    let mut input = unit.prepare_native_compilation().map_err(|error| {
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
        .aot_object_data(&assembly)
        .map_err(|error| AotError::contract("aot.compile.data", error.to_string()))?;
    runmat_native_codegen::aot::emit_relocatable_object_with_data(
        &assembly,
        options.optimization,
        data,
    )
    .map_err(|error| AotError::contract("aot.compile.object", error.to_string()))
}
