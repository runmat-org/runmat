use std::collections::BTreeMap;
use std::rc::Rc;

use runmat_types::FunctionId;

use crate::ExecutableUnit;

pub(crate) fn capture_liveness_admission(
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<runmat_native_codegen::NativeLivenessAdmission, runmat_runtime::RuntimeError> {
    runmat_native_codegen::NativeLivenessAdmission::capture(runtime).map_err(|source| {
        runmat_runtime::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            "RunMat:NativeLivenessAdmission",
            "native liveness metadata admission failed",
            source,
        )
    })
}

pub(super) struct CompiledGenericUnit {
    pub executor: Rc<runmat_native_executor::NativeExecutor>,
    pub entrypoint: FunctionId,
    pub compilation: runmat_jit::entry::NativeCompilationEvidence,
    #[cfg(test)]
    pub safepoints: BTreeMap<FunctionId, Vec<runmat_native_codegen::NativeSafepointId>>,
    #[cfg(test)]
    pub safepoint_locations: BTreeMap<
        FunctionId,
        Vec<(
            runmat_native_codegen::NativeSafepointId,
            runmat_native_codegen::NativeSourceLocation,
        )>,
    >,
}

pub(super) struct PreparedGenericUnit {
    native: crate::NativeCompilationInput,
    entrypoint: FunctionId,
    specialization: Option<runmat_jit::tiering::RepresentationProfile>,
    prepare_ns: u64,
    optimization: runmat_jit::CompilerOptimization,
    liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
}

pub(super) struct BackgroundCompiledGenericUnit {
    pub executor: runmat_native_executor::NativeExecutor,
    pub entrypoint: FunctionId,
    pub compilation: runmat_jit::entry::NativeCompilationEvidence,
    #[cfg(test)]
    pub safepoints: BTreeMap<FunctionId, Vec<runmat_native_codegen::NativeSafepointId>>,
    #[cfg(test)]
    pub safepoint_locations: BTreeMap<
        FunctionId,
        Vec<(
            runmat_native_codegen::NativeSafepointId,
            runmat_native_codegen::NativeSourceLocation,
        )>,
    >,
}

#[cfg(test)]
pub(super) fn compile(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
) -> Result<CompiledGenericUnit, runmat_runtime::RuntimeError> {
    compile_with_policy(
        unit,
        preferred_function,
        runmat_jit::CompilerOptimization::TierDefault,
        runmat_native_codegen::NativeLivenessAdmission::AuthorizedUnbounded,
    )
}

pub(super) fn compile_with_policy(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
    optimization: runmat_jit::CompilerOptimization,
    liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
) -> Result<CompiledGenericUnit, runmat_runtime::RuntimeError> {
    let compiled = compile_prepared(prepare_with_policy(
        unit,
        preferred_function,
        None,
        optimization,
        liveness_admission,
    )?)?;
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

pub(super) fn prepare_with_policy(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
    specialization: Option<runmat_jit::tiering::RepresentationProfile>,
    optimization: runmat_jit::CompilerOptimization,
    liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
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
        optimization,
        liveness_admission,
    })
}

pub(super) fn compile_prepared(
    prepared: PreparedGenericUnit,
) -> Result<BackgroundCompiledGenericUnit, runmat_runtime::RuntimeError> {
    let started = std::time::Instant::now();
    let program_capture = prepared.native.program_capture().to_vec();
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
    let executor = runmat_jit::GenericCompiler::compile_executor_with_policy(
        assembly,
        Some(program_capture),
        interpreter_resume_points,
        coverage_sites,
        prepared.specialization,
        prepared.optimization,
        prepared.liveness_admission,
    )
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
