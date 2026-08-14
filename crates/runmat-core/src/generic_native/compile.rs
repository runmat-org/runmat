use std::rc::Rc;

use runmat_types::ProgramFunctionId;

use crate::ExecutableUnit;

pub(super) struct CompiledGenericUnit {
    pub executor: Rc<runmat_jit::GenericExecutor>,
    pub entrypoint: ProgramFunctionId,
}

pub(super) fn compile(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
) -> Result<CompiledGenericUnit, runmat_runtime::RuntimeError> {
    let envelope = unit
        .portable_envelope_for(preferred_function)
        .map_err(|error| super::error::stage("NativeProduct", error))?;
    let binding_names = unit.binding_names();
    let assembly =
        runmat_native_codegen::lower_executable(runmat_native_codegen::NativeLoweringInput {
            mir: unit.mir(),
            analysis: unit.analysis(),
            manifest: &envelope.manifest,
            binding_names: Some(&binding_names),
            target: runmat_native_codegen::NativeTarget::current(),
        })
        .map_err(|error| super::error::stage("NativeLowering", error))?;
    let program_capture = serde_json::to_vec(unit.functions()).map_err(|error| {
        super::error::stage(
            "NativeProduct",
            format!("failed to capture native async program: {error}"),
        )
    })?;
    let executor =
        runmat_jit::GenericExecutor::compile_with_program_capture(assembly, Some(program_capture))
            .map_err(super::error::from_jit_error)?;
    Ok(CompiledGenericUnit {
        executor: Rc::new(executor),
        entrypoint: envelope.manifest.identity.entrypoint_function,
    })
}
