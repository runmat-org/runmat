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
    mir: runmat_mir::MirAssembly,
    analysis: runmat_mir::analysis::AnalysisStore,
    manifest: runmat_execution::ExecutableUnitManifest,
    binding_names: BTreeMap<runmat_types::BindingId, String>,
    program_capture: Vec<u8>,
    interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
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
    let envelope = unit
        .portable_envelope_for(preferred_function)
        .map_err(|error| super::error::stage("NativeProduct", error))?;
    let binding_names = unit.binding_names();
    let program_capture = serde_json::to_vec(unit.functions()).map_err(|error| {
        super::error::stage(
            "NativeProduct",
            format!("failed to capture native async program: {error}"),
        )
    })?;
    let mut interpreter_resume_points = unit
        .vm_layout()
        .functions
        .values()
        .flat_map(|function| function.resume_points.iter())
        .map(|(point, pc)| {
            u64::try_from(*pc).map(|pc| (*point, pc)).map_err(|_| {
                super::error::stage("NativeProduct", "bytecode resume PC exceeds native ABI")
            })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    for function in unit.functions().functions.values() {
        for (point, pc) in &function.resume_points {
            let pc = u64::try_from(*pc).map_err(|_| {
                super::error::stage("NativeProduct", "bytecode resume PC exceeds native ABI")
            })?;
            interpreter_resume_points.insert(*point, pc);
        }
    }
    let entrypoint = envelope.manifest.identity.entrypoint_function;
    Ok(PreparedGenericUnit {
        mir: unit.mir().clone(),
        analysis: unit.analysis().clone(),
        manifest: envelope.manifest,
        binding_names,
        program_capture,
        interpreter_resume_points,
        entrypoint,
        specialization,
    })
}

pub(super) fn compile_prepared(
    prepared: PreparedGenericUnit,
) -> Result<BackgroundCompiledGenericUnit, runmat_runtime::RuntimeError> {
    let assembly =
        runmat_native_codegen::lower_executable(runmat_native_codegen::NativeLoweringInput {
            mir: &prepared.mir,
            analysis: &prepared.analysis,
            manifest: &prepared.manifest,
            binding_names: Some(&prepared.binding_names),
            target: runmat_native_codegen::NativeTarget::current(),
        })
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
            Some(prepared.program_capture),
            prepared.interpreter_resume_points,
            profile,
        ),
        None => runmat_jit::GenericExecutor::compile_with_resume_points(
            assembly,
            Some(prepared.program_capture),
            prepared.interpreter_resume_points,
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
