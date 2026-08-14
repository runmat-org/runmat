#[cfg(test)]
use std::collections::BTreeMap;
use std::rc::Rc;

use runmat_types::ProgramFunctionId;

use crate::ExecutableUnit;

pub(super) struct CompiledGenericUnit {
    pub executor: Rc<runmat_jit::GenericExecutor>,
    pub entrypoint: ProgramFunctionId,
    #[cfg(test)]
    pub safepoints: BTreeMap<ProgramFunctionId, Vec<runmat_native_codegen::NativeSafepointId>>,
}

pub(super) struct PreparedGenericUnit {
    native: crate::NativeCompilationInput,
    entrypoint: ProgramFunctionId,
    specialization: Option<runmat_jit::tiering::RepresentationProfile>,
}

pub(super) struct BackgroundCompiledGenericUnit {
    pub executor: runmat_jit::GenericExecutor,
    pub entrypoint: ProgramFunctionId,
    #[cfg(test)]
    pub safepoints: BTreeMap<ProgramFunctionId, Vec<runmat_native_codegen::NativeSafepointId>>,
}

pub(super) fn compile(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
) -> Result<CompiledGenericUnit, runmat_runtime::RuntimeError> {
    let compiled = compile_prepared(prepare(unit, preferred_function, None)?)?;
    Ok(CompiledGenericUnit {
        executor: Rc::new(compiled.executor),
        entrypoint: compiled.entrypoint,
        #[cfg(test)]
        safepoints: compiled.safepoints,
    })
}

pub(super) fn prepare(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
    specialization: Option<runmat_jit::tiering::RepresentationProfile>,
) -> Result<PreparedGenericUnit, runmat_runtime::RuntimeError> {
    let native = unit.prepare_native_compilation_for(preferred_function)?;
    let entrypoint = native.entrypoint();
    Ok(PreparedGenericUnit {
        native,
        entrypoint,
        specialization,
    })
}

pub(super) fn compile_prepared(
    prepared: PreparedGenericUnit,
) -> Result<BackgroundCompiledGenericUnit, runmat_runtime::RuntimeError> {
    let program_capture = prepared.native.program_capture().to_vec();
    let interpreter_resume_points = prepared.native.interpreter_resume_points().clone();
    let assembly = prepared
        .native
        .lower(runmat_native_codegen::NativeTarget::current())
        .map_err(|error| super::error::stage("NativeLowering", error))?;
    #[cfg(test)]
    let safepoints = assembly
        .functions
        .iter()
        .map(|function| {
            let points = function
                .blocks
                .iter()
                .flat_map(|block| {
                    block
                        .instructions
                        .iter()
                        .filter_map(|instruction| instruction.safepoint)
                        .chain(block.terminator.safepoint)
                })
                .collect();
            (function.id, points)
        })
        .collect();
    let executor = match prepared.specialization {
        Some(profile) => runmat_jit::GenericExecutor::compile_specialized_with_resume_points(
            assembly,
            Some(program_capture),
            interpreter_resume_points,
            profile,
        ),
        None => runmat_jit::GenericExecutor::compile_with_resume_points(
            assembly,
            Some(program_capture),
            interpreter_resume_points,
        ),
    }
    .map_err(super::error::from_jit_error)?;
    Ok(BackgroundCompiledGenericUnit {
        executor,
        entrypoint: prepared.entrypoint,
        #[cfg(test)]
        safepoints,
    })
}
