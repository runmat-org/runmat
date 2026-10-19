use std::collections::BTreeMap;
use std::rc::Rc;

use runmat_types::ProgramFunctionId;

use crate::ExecutableUnit;

pub(super) struct CompiledGenericUnit {
    pub executor: Rc<runmat_native_executor::NativeExecutor>,
    pub entrypoint: ProgramFunctionId,
    pub compilation: runmat_jit::entry::NativeCompilationEvidence,
    #[cfg(test)]
    pub safepoints: BTreeMap<ProgramFunctionId, Vec<runmat_native_codegen::NativeSafepointId>>,
    #[cfg(test)]
    pub safepoint_locations: BTreeMap<
        ProgramFunctionId,
        Vec<(
            runmat_native_codegen::NativeSafepointId,
            runmat_native_codegen::NativeSourceLocation,
        )>,
    >,
}

pub(super) struct PreparedGenericUnit {
    native: crate::NativeCompilationInput,
    entrypoint: ProgramFunctionId,
    specialization: Option<runmat_jit::tiering::RepresentationProfile>,
    prepare_ns: u64,
}

pub(super) struct BackgroundCompiledGenericUnit {
    pub executor: runmat_native_executor::NativeExecutor,
    pub entrypoint: ProgramFunctionId,
    pub compilation: runmat_jit::entry::NativeCompilationEvidence,
    #[cfg(test)]
    pub safepoints: BTreeMap<ProgramFunctionId, Vec<runmat_native_codegen::NativeSafepointId>>,
    #[cfg(test)]
    pub safepoint_locations: BTreeMap<
        ProgramFunctionId,
        Vec<(
            runmat_native_codegen::NativeSafepointId,
            runmat_native_codegen::NativeSourceLocation,
        )>,
    >,
}

pub(super) fn compile(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
) -> Result<CompiledGenericUnit, runmat_runtime::RuntimeError> {
    let compiled = compile_prepared(prepare(unit, preferred_function, None)?)?;
    Ok(CompiledGenericUnit {
        executor: Rc::new(compiled.executor),
        entrypoint: compiled.entrypoint,
        compilation: compiled.compilation,
        #[cfg(test)]
        safepoints: compiled.safepoints,
        #[cfg(test)]
        safepoint_locations: compiled.safepoint_locations,
    })
}

pub(super) fn prepare(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
    specialization: Option<runmat_jit::tiering::RepresentationProfile>,
) -> Result<PreparedGenericUnit, runmat_runtime::RuntimeError> {
    let started = std::time::Instant::now();
    let native = unit.prepare_native_compilation_for(preferred_function)?;
    let entrypoint = native.entrypoint();
    let prepare_ns = runmat_time::duration_ns_saturating(started.elapsed());
    Ok(PreparedGenericUnit {
        native,
        entrypoint,
        specialization,
        prepare_ns,
    })
}

pub(super) fn compile_prepared(
    prepared: PreparedGenericUnit,
) -> Result<BackgroundCompiledGenericUnit, runmat_runtime::RuntimeError> {
    let started = std::time::Instant::now();
    let interpreter_resume_points = prepared.native.interpreter_resume_points().clone();
    let point_coverage = prepared.native.coverage_sites().clone();
    let assembly = prepared
        .native
        .lower(runmat_native_codegen::NativeTarget::current())
        .map_err(|error| super::error::stage("NativeLowering", error))?;
    let mut assigned_points = std::collections::BTreeSet::new();
    let mut coverage_sites = BTreeMap::new();
    for site in assembly
        .functions
        .iter()
        .flat_map(|function| function.expected_sites.iter())
    {
        if !assigned_points.insert(site.point) {
            continue;
        }
        if let Some(sites) = point_coverage.get(&site.point) {
            coverage_sites.insert(site.clone(), sites.clone());
        }
    }
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
    #[cfg(test)]
    let safepoint_locations = assembly
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
                        .filter_map(|instruction| {
                            instruction
                                .safepoint
                                .map(|id| (id, instruction.source.clone()))
                        })
                        .chain(
                            block
                                .terminator
                                .safepoint
                                .map(|id| (id, block.terminator.source.clone())),
                        )
                })
                .collect();
            (function.id, points)
        })
        .collect();
    let executor = match prepared.specialization {
        Some(profile) => runmat_jit::GenericCompiler::compile_specialized_executor_with_metadata(
            assembly,
            interpreter_resume_points,
            coverage_sites,
            profile,
        ),
        None => runmat_jit::GenericCompiler::compile_executor_with_metadata(
            assembly,
            interpreter_resume_points,
            coverage_sites,
        ),
    }
    .map_err(super::error::from_jit_error)?;
    Ok(BackgroundCompiledGenericUnit {
        executor,
        entrypoint: prepared.entrypoint,
        compilation: runmat_jit::entry::NativeCompilationEvidence {
            prepare_ns: prepared.prepare_ns,
            lower_codegen_bind_ns: runmat_time::duration_ns_saturating(started.elapsed()),
        },
        #[cfg(test)]
        safepoints,
        #[cfg(test)]
        safepoint_locations,
    })
}
