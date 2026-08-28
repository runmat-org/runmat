use std::collections::BTreeMap;

use runmat_native_executor::{NativeExecutor, NativeExecutorOptions, RepresentationProfile};
use runmat_types::ProgramPointId;

use super::GenericCompiler;
use crate::JitResult;

impl GenericCompiler {
    pub fn compile_executor_with_resume_points(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<ProgramPointId, u64>,
    ) -> JitResult<NativeExecutor> {
        Self::compile_executor_product(
            assembly,
            program_capture,
            interpreter_resume_points,
            BTreeMap::new(),
            None,
        )
    }

    pub fn compile_executor_with_metadata(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<ProgramPointId, u64>,
        coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
    ) -> JitResult<NativeExecutor> {
        Self::compile_executor_product(
            assembly,
            program_capture,
            interpreter_resume_points,
            coverage_sites,
            None,
        )
    }

    pub fn compile_specialized_executor_with_resume_points(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<ProgramPointId, u64>,
        profile: RepresentationProfile,
    ) -> JitResult<NativeExecutor> {
        Self::compile_executor_product(
            assembly,
            program_capture,
            interpreter_resume_points,
            BTreeMap::new(),
            Some(profile),
        )
    }

    pub fn compile_specialized_executor_with_metadata(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<ProgramPointId, u64>,
        coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
        profile: RepresentationProfile,
    ) -> JitResult<NativeExecutor> {
        Self::compile_executor_product(
            assembly,
            program_capture,
            interpreter_resume_points,
            coverage_sites,
            Some(profile),
        )
    }

    fn compile_executor_product(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<ProgramPointId, u64>,
        coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
        entry_profile: Option<RepresentationProfile>,
    ) -> JitResult<NativeExecutor> {
        let compile_started = std::time::Instant::now();
        let executable = if entry_profile.is_some() {
            Self::compile_specialized(&assembly)?
        } else {
            Self::compile(&assembly)?
        };
        let compile_duration_ns = runmat_time::duration_ns_saturating(compile_started.elapsed());
        NativeExecutor::bind(
            assembly,
            executable,
            NativeExecutorOptions {
                program_capture,
                interpreter_resume_points,
                coverage_sites,
                entry_profile,
                compile_duration_ns,
            },
        )
        .map_err(Into::into)
    }
}
