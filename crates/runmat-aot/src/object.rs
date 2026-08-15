use runmat_native_codegen::aot::{NativeOptimization, RelocatableNativeObject};

use crate::{AotError, AotResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeObjectOptions {
    pub optimization: NativeOptimization,
}

impl Default for NativeObjectOptions {
    fn default() -> Self {
        Self {
            optimization: NativeOptimization::Speed,
        }
    }
}

pub fn emit_native_object(
    unit: &runmat_core::ExecutableUnit,
    options: NativeObjectOptions,
) -> AotResult<RelocatableNativeObject> {
    let input = unit.prepare_native_compilation().map_err(|error| {
        AotError::contract(
            "aot.compile.input",
            format!("failed to prepare canonical native input: {error}"),
        )
    })?;
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
