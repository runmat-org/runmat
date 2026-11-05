use std::collections::BTreeMap;

use runmat_native_executor::{NativeExecutor, NativeExecutorOptions, RepresentationProfile};
use runmat_types::ProgramPointId;

use super::{CompilerOptimization, GenericCompiler};
use crate::JitResult;

impl GenericCompiler {
    pub fn compile_executor_with_resume_points(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<ProgramPointId, u64>,
        liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
    ) -> JitResult<NativeExecutor> {
        Self::compile_executor_product(
            assembly,
            program_capture,
            interpreter_resume_points,
            BTreeMap::new(),
            None,
            liveness_admission,
        )
    }

    pub fn compile_executor_with_metadata(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<ProgramPointId, u64>,
        coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
        liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
    ) -> JitResult<NativeExecutor> {
        Self::compile_executor_product(
            assembly,
            program_capture,
            interpreter_resume_points,
            coverage_sites,
            None,
            liveness_admission,
        )
    }

    pub fn compile_specialized_executor_with_resume_points(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<ProgramPointId, u64>,
        profile: RepresentationProfile,
        liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
    ) -> JitResult<NativeExecutor> {
        Self::compile_executor_product(
            assembly,
            program_capture,
            interpreter_resume_points,
            BTreeMap::new(),
            Some(profile),
            liveness_admission,
        )
    }

    pub fn compile_specialized_executor_with_metadata(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<ProgramPointId, u64>,
        coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
        profile: RepresentationProfile,
        liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
    ) -> JitResult<NativeExecutor> {
        Self::compile_executor_product(
            assembly,
            program_capture,
            interpreter_resume_points,
            coverage_sites,
            Some(profile),
            liveness_admission,
        )
    }

    fn compile_executor_product(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<ProgramPointId, u64>,
        coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
        entry_profile: Option<RepresentationProfile>,
        liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
    ) -> JitResult<NativeExecutor> {
        Self::compile_executor_with_policy(
            assembly,
            program_capture,
            interpreter_resume_points,
            coverage_sites,
            entry_profile,
            CompilerOptimization::TierDefault,
            liveness_admission,
        )
    }

    pub fn compile_executor_with_policy(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<ProgramPointId, u64>,
        coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
        entry_profile: Option<RepresentationProfile>,
        optimization: CompilerOptimization,
        liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
    ) -> JitResult<NativeExecutor> {
        let compile_started = std::time::Instant::now();
        let executable = if optimization != CompilerOptimization::TierDefault {
            Self::compile_with_policy(&assembly, optimization)?
        } else if entry_profile.is_some() {
            Self::compile_specialized(&assembly)?
        } else {
            Self::compile(&assembly)?
        };
        let compile_duration_ns = runmat_time::duration_ns_saturating(compile_started.elapsed());
        NativeExecutor::bind(
            assembly,
            executable,
            NativeExecutorOptions {
                selected_clock_invocation: None,
                selected_local_invocation: None,
                program_capture,
                interpreter_resume_points,
                coverage_sites,
                entry_profile,
                compile_duration_ns,
            },
            liveness_admission,
        )
        .map_err(Into::into)
    }
}
