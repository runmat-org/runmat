use runmat_runtime::context::RuntimeContext;
use runmat_types::ProgramFunctionId;
use runmat_value::Value;
use std::{collections::BTreeMap, sync::Arc};

use crate::deopt::DeoptimizationPolicy;
use crate::osr::OsrTarget;
use crate::{CompiledExecutable, GenericCompiler, JitError, JitResult};

use super::invocation::{GenericInvocation, GenericInvocationStep};
use super::state::{HostState, HostStateInput};

pub struct GenericExecutor {
    functions: Arc<Vec<runmat_native_codegen::NativeFunction>>,
    compiled: CompiledExecutable,
    program_capture: Option<Vec<u8>>,
    interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    pub(super) regions: Vec<runmat_types::RegionContract>,
    pub(super) compile_duration_ns: u64,
    entry_profile: Option<crate::tiering::RepresentationProfile>,
    optimized_regions: Arc<Vec<crate::region::OptimizedRegionPlan>>,
}

#[derive(Debug, PartialEq)]
pub struct GenericExecution {
    pub outputs: Vec<Value>,
    pub captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
    pub loop_backedges: BTreeMap<runmat_types::ProgramPointId, u64>,
    pub osr_entry: Option<runmat_types::ProgramPointId>,
    pub vectorized_regions: u64,
}

impl GenericExecutor {
    pub fn compile(assembly: runmat_native_codegen::NativeAssembly) -> JitResult<Self> {
        Self::compile_with_program_capture(assembly, None)
    }

    pub fn compile_with_program_capture(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
    ) -> JitResult<Self> {
        Self::compile_with_resume_points(assembly, program_capture, BTreeMap::new())
    }

    pub fn compile_with_resume_points(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    ) -> JitResult<Self> {
        Self::compile_product(assembly, program_capture, interpreter_resume_points, None)
    }

    pub fn compile_specialized_with_resume_points(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
        profile: crate::tiering::RepresentationProfile,
    ) -> JitResult<Self> {
        Self::compile_product(
            assembly,
            program_capture,
            interpreter_resume_points,
            Some(profile),
        )
    }

    fn compile_product(
        assembly: runmat_native_codegen::NativeAssembly,
        program_capture: Option<Vec<u8>>,
        interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
        entry_profile: Option<crate::tiering::RepresentationProfile>,
    ) -> JitResult<Self> {
        let regions = assembly.requirements.regions.clone();
        let compile_started = std::time::Instant::now();
        let optimized_regions = if entry_profile.is_some() {
            crate::region::derive_plans(&assembly.functions, &regions)
        } else {
            Vec::new()
        };
        let compiled = if entry_profile.is_some() {
            GenericCompiler::compile_specialized(&assembly)?
        } else {
            GenericCompiler::compile(&assembly)?
        };
        let compile_duration_ns = runmat_time::duration_ns_saturating(compile_started.elapsed());
        Ok(Self {
            functions: Arc::new(assembly.functions),
            compiled,
            program_capture,
            interpreter_resume_points,
            regions,
            compile_duration_ns,
            entry_profile,
            optimized_regions: Arc::new(optimized_regions),
        })
    }

    pub fn retained_code_bytes(&self) -> u64 {
        self.compiled.retained_code_bytes()
    }

    pub fn entry_profile(&self) -> Option<&crate::tiering::RepresentationProfile> {
        self.entry_profile.as_ref()
    }

    pub fn optimized_region_count(&self) -> usize {
        self.optimized_regions.len()
    }

    pub(crate) fn compiled_entrypoint(
        &self,
        function: ProgramFunctionId,
    ) -> JitResult<runmat_runtime::native::NativeEntryPoint> {
        self.compiled.entrypoint(function)
    }

    pub fn invoke(
        &self,
        function: ProgramFunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> JitResult<GenericExecution> {
        let mut invocation = self.begin(function, arguments, requested_outputs, runtime)?;
        loop {
            match invocation.advance()? {
                GenericInvocationStep::Completed(execution) => return Ok(execution),
                GenericInvocationStep::Suspended { .. } => {
                    return Err(JitError::UnsupportedExit(
                        runmat_runtime::native::NativeExitKind::SUSPENDED.0,
                    ))
                }
                GenericInvocationStep::Deoptimized { target, .. }
                    if target == runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE =>
                {
                    invocation.resume_deoptimization()?
                }
                GenericInvocationStep::Deoptimized { target, .. }
                    if target == runmat_runtime::native::NativeResumeKind::OPTIMIZED_NATIVE =>
                {
                    invocation.resume_optimization()?
                }
                GenericInvocationStep::Deoptimized { .. } => {
                    return Err(JitError::UnsupportedExit(
                        runmat_runtime::native::NativeExitKind::DEOPTIMIZED.0,
                    ))
                }
            }
        }
    }

    pub async fn invoke_async(
        &self,
        function: ProgramFunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> JitResult<GenericExecution> {
        self.invoke_async_with_captures(function, Vec::new(), arguments, requested_outputs, runtime)
            .await
    }

    pub async fn invoke_async_with_captures(
        &self,
        function: ProgramFunctionId,
        captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> JitResult<GenericExecution> {
        let mut invocation =
            self.begin_with_captures(function, captures, arguments, requested_outputs, runtime)?;
        loop {
            match invocation.advance()? {
                GenericInvocationStep::Completed(execution) => return Ok(execution),
                GenericInvocationStep::Suspended {
                    continuation,
                    generation,
                } => {
                    invocation
                        .resume_suspension(continuation, generation)
                        .await?;
                }
                GenericInvocationStep::Deoptimized { target, .. }
                    if target == runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE =>
                {
                    invocation.resume_deoptimization()?;
                }
                GenericInvocationStep::Deoptimized { target, .. }
                    if target == runmat_runtime::native::NativeResumeKind::OPTIMIZED_NATIVE =>
                {
                    invocation.resume_optimization()?;
                }
                GenericInvocationStep::Deoptimized { .. } => {
                    return Err(JitError::UnsupportedExit(
                        runmat_runtime::native::NativeExitKind::DEOPTIMIZED.0,
                    ));
                }
            }
        }
    }

    pub fn begin(
        &self,
        function: ProgramFunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> JitResult<GenericInvocation> {
        self.begin_with_captures(function, Vec::new(), arguments, requested_outputs, runtime)
    }

    pub fn begin_with_captures(
        &self,
        function: ProgramFunctionId,
        captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> JitResult<GenericInvocation> {
        self.begin_with_deoptimization(
            function,
            captures,
            arguments,
            requested_outputs,
            runtime,
            DeoptimizationPolicy::default(),
        )
    }

    pub fn begin_with_deoptimization(
        &self,
        function: ProgramFunctionId,
        captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
        deoptimization: DeoptimizationPolicy,
    ) -> JitResult<GenericInvocation> {
        self.begin_with_deoptimization_and_osr(
            function,
            captures,
            arguments,
            requested_outputs,
            runtime,
            deoptimization,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn begin_with_deoptimization_and_osr(
        &self,
        function: ProgramFunctionId,
        captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
        deoptimization: DeoptimizationPolicy,
        osr_target: Option<OsrTarget>,
    ) -> JitResult<GenericInvocation> {
        self.validate_entry_profile(&arguments)?;
        if let Some(target) = &osr_target {
            target.executor().validate_entry_profile(&arguments)?;
        }
        let function_ir = self
            .functions
            .iter()
            .find(|candidate| candidate.id == function)
            .cloned()
            .ok_or_else(|| {
                JitError::Host(format!("native function {} is unavailable", function.0))
            })?;
        if let Some(target) = &osr_target {
            let point = target.point();
            if point.function != function
                || !function_ir.blocks.iter().any(|block| {
                    block.terminator.site.point == point
                        && matches!(
                            block.terminator.kind,
                            runmat_native_codegen::NativeTerminatorKind::For { .. }
                        )
                })
            {
                return Err(JitError::Host(
                    "OSR target is not an exact for-loop header in this function".into(),
                ));
            }
        }
        let requested_outputs = u32::try_from(requested_outputs)
            .map_err(|_| JitError::Host("requested output count exceeds native ABI".into()))?;
        let (state, argument_refs) = HostState::new(HostStateInput {
            function: function_ir,
            arguments,
            requested_outputs: requested_outputs as usize,
            runtime,
            program_capture: self.program_capture.clone(),
            functions: Arc::clone(&self.functions),
            captures,
            deoptimization,
            interpreter_resume_points: self.interpreter_resume_points.clone(),
            osr_point: osr_target.as_ref().map(OsrTarget::point),
            optimized_regions: Arc::clone(&self.optimized_regions),
        })?;
        let resume = runmat_runtime::native::NativeResumeState {
            function: function.0,
            block: state.function.entry.0,
            phase: runmat_runtime::native::NativeSitePhase::RVALUE.0,
            local_count: state.locals.len() as u32,
            source: runmat_runtime::native::NativeSourceLocation {
                source: state.function.source.0,
                ..runmat_runtime::native::NativeSourceLocation::default()
            },
            ..runmat_runtime::native::NativeResumeState::default()
        };
        let entrypoint = self.compiled.entrypoint(function)?;
        let osr_target = if let Some(target) = osr_target {
            let entrypoint = target.entrypoint()?;
            Some((target, entrypoint))
        } else {
            None
        };
        Ok(GenericInvocation::new(
            state,
            entrypoint,
            argument_refs,
            requested_outputs,
            resume,
            osr_target,
        ))
    }

    fn validate_entry_profile(&self, arguments: &[Value]) -> JitResult<()> {
        let Some(profile) = &self.entry_profile else {
            return Ok(());
        };
        let actual = arguments
            .iter()
            .map(runmat_runtime::value_fact::value_fact)
            .collect::<Vec<_>>();
        let actual = crate::tiering::RepresentationProfile::from_facts(actual, usize::MAX)
            .map_err(|error| JitError::Host(error.into()))?;
        if actual.digest != profile.digest || actual.facts != profile.facts {
            return Err(JitError::Host(
                "specialized native entry representation guard failed".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::GenericExecutor;

    fn assert_send<T: Send>() {}

    #[test]
    fn compiled_executor_can_move_from_a_background_compiler_to_its_session_owner() {
        assert_send::<GenericExecutor>();
    }
}
