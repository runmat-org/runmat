use runmat_runtime::context::RuntimeContext;
use runmat_types::ProgramFunctionId;
use runmat_value::Value;
use std::{collections::BTreeMap, sync::Arc};

use crate::deopt::DeoptimizationPolicy;
use crate::osr::OsrTarget;
use crate::{NativeExecutable, NativeExecutorError, NativeExecutorResult};

use super::invocation::{NativeInvocation, NativeInvocationStep};
use super::state::{HostState, HostStateInput};

pub struct NativeExecutor {
    functions: Arc<Vec<runmat_native_codegen::NativeFunction>>,
    compiled: NativeExecutable,
    program_capture: Option<Vec<u8>>,
    interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    pub(super) regions: Vec<runmat_types::RegionContract>,
    pub(super) compile_duration_ns: u64,
    entry_profile: Option<crate::RepresentationProfile>,
    optimized_regions: Arc<Vec<crate::region::OptimizedRegionPlan>>,
}

#[derive(Default)]
pub struct NativeExecutorOptions {
    pub program_capture: Option<Vec<u8>>,
    pub interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    pub entry_profile: Option<crate::RepresentationProfile>,
    pub compile_duration_ns: u64,
}

pub struct NativeInvocationRequest {
    pub function: ProgramFunctionId,
    pub captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
    pub runtime: RuntimeContext,
    pub deoptimization: DeoptimizationPolicy,
    pub osr_target: Option<OsrTarget>,
    pub workspace: Option<super::workspace::NativeWorkspaceInput>,
}

#[derive(Debug, PartialEq)]
pub struct NativeExecution {
    pub outputs: Vec<Value>,
    pub captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
    pub loop_backedges: BTreeMap<runmat_types::ProgramPointId, u64>,
    pub osr_entry: Option<runmat_types::ProgramPointId>,
    pub vectorized_regions: u64,
    pub workspace: Option<super::workspace::NativeWorkspaceSnapshot>,
    pub expression: Option<Value>,
}

impl NativeExecutor {
    /// Bind verified Native IR to an executable with the same exact function set.
    pub fn bind(
        assembly: runmat_native_codegen::NativeAssembly,
        executable: NativeExecutable,
        options: NativeExecutorOptions,
    ) -> NativeExecutorResult<Self> {
        assembly.verify()?;
        let ir_functions = assembly
            .functions
            .iter()
            .map(|function| function.id)
            .collect::<std::collections::BTreeSet<_>>();
        let executable_functions = executable
            .function_ids()
            .collect::<std::collections::BTreeSet<_>>();
        if executable_functions != ir_functions {
            return Err(NativeExecutorError::Executable(
                "native executable entrypoints do not exactly match the verified Native IR product"
                    .into(),
            ));
        }
        let regions = assembly.requirements.regions.clone();
        let optimized_regions = if options.entry_profile.is_some() {
            crate::region::derive_plans(&assembly.functions, &regions)
        } else {
            Vec::new()
        };
        Ok(Self {
            functions: Arc::new(assembly.functions),
            compiled: executable,
            program_capture: options.program_capture,
            interpreter_resume_points: options.interpreter_resume_points,
            regions,
            compile_duration_ns: options.compile_duration_ns,
            entry_profile: options.entry_profile,
            optimized_regions: Arc::new(optimized_regions),
        })
    }

    pub fn retained_code_bytes(&self) -> u64 {
        self.compiled.retained_code_bytes()
    }

    pub fn entry_profile(&self) -> Option<&crate::RepresentationProfile> {
        self.entry_profile.as_ref()
    }

    pub fn optimized_region_count(&self) -> usize {
        self.optimized_regions.len()
    }

    pub(crate) fn compiled_entrypoint(
        &self,
        function: ProgramFunctionId,
    ) -> NativeExecutorResult<runmat_runtime::native::NativeEntryPoint> {
        self.compiled.entrypoint(function)
    }

    pub fn invoke(
        &self,
        function: ProgramFunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> NativeExecutorResult<NativeExecution> {
        let mut invocation = self.begin(function, arguments, requested_outputs, runtime)?;
        loop {
            match invocation.advance()? {
                NativeInvocationStep::Completed(execution) => return Ok(execution),
                NativeInvocationStep::Suspended { .. } => {
                    return Err(NativeExecutorError::UnsupportedExit(
                        runmat_runtime::native::NativeExitKind::SUSPENDED.0,
                    ))
                }
                NativeInvocationStep::Deoptimized { target, .. }
                    if target == runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE =>
                {
                    invocation.resume_deoptimization()?
                }
                NativeInvocationStep::Deoptimized { target, .. }
                    if target == runmat_runtime::native::NativeResumeKind::OPTIMIZED_NATIVE =>
                {
                    invocation.resume_optimization()?
                }
                NativeInvocationStep::Deoptimized { .. } => {
                    return Err(NativeExecutorError::UnsupportedExit(
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
    ) -> NativeExecutorResult<NativeExecution> {
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
    ) -> NativeExecutorResult<NativeExecution> {
        let mut invocation =
            self.begin_with_captures(function, captures, arguments, requested_outputs, runtime)?;
        loop {
            match invocation.advance()? {
                NativeInvocationStep::Completed(execution) => return Ok(execution),
                NativeInvocationStep::Suspended {
                    continuation,
                    generation,
                } => {
                    invocation
                        .resume_suspension(continuation, generation)
                        .await?;
                }
                NativeInvocationStep::Deoptimized { target, .. }
                    if target == runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE =>
                {
                    invocation.resume_deoptimization()?;
                }
                NativeInvocationStep::Deoptimized { target, .. }
                    if target == runmat_runtime::native::NativeResumeKind::OPTIMIZED_NATIVE =>
                {
                    invocation.resume_optimization()?;
                }
                NativeInvocationStep::Deoptimized { .. } => {
                    return Err(NativeExecutorError::UnsupportedExit(
                        runmat_runtime::native::NativeExitKind::DEOPTIMIZED.0,
                    ));
                }
            }
        }
    }

    pub async fn invoke_workspace_async(
        &self,
        function: ProgramFunctionId,
        workspace: super::workspace::NativeWorkspaceInput,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> NativeExecutorResult<NativeExecution> {
        let mut invocation = self.begin_request(NativeInvocationRequest {
            function,
            captures: Vec::new(),
            arguments: Vec::new(),
            requested_outputs,
            runtime,
            deoptimization: DeoptimizationPolicy::default(),
            osr_target: None,
            workspace: Some(workspace),
        })?;
        loop {
            match invocation.advance()? {
                NativeInvocationStep::Completed(execution) => return Ok(execution),
                NativeInvocationStep::Suspended {
                    continuation,
                    generation,
                } => {
                    invocation
                        .resume_suspension(continuation, generation)
                        .await?;
                }
                NativeInvocationStep::Deoptimized { target, .. }
                    if target == runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE =>
                {
                    invocation.resume_deoptimization()?;
                }
                NativeInvocationStep::Deoptimized { .. } => {
                    return Err(NativeExecutorError::UnsupportedExit(
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
    ) -> NativeExecutorResult<NativeInvocation> {
        self.begin_with_captures(function, Vec::new(), arguments, requested_outputs, runtime)
    }

    pub fn begin_with_captures(
        &self,
        function: ProgramFunctionId,
        captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> NativeExecutorResult<NativeInvocation> {
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
    ) -> NativeExecutorResult<NativeInvocation> {
        self.begin_request(NativeInvocationRequest {
            function,
            captures,
            arguments,
            requested_outputs,
            runtime,
            deoptimization,
            osr_target: None,
            workspace: None,
        })
    }

    pub fn begin_request(
        &self,
        request: NativeInvocationRequest,
    ) -> NativeExecutorResult<NativeInvocation> {
        let NativeInvocationRequest {
            function,
            captures,
            arguments,
            requested_outputs,
            runtime,
            deoptimization,
            osr_target,
            workspace,
        } = request;
        let mut profile_values = arguments.clone();
        if let Some(workspace) = &workspace {
            profile_values.extend(workspace.profile_values());
        }
        self.validate_entry_profile(&profile_values)?;
        if let Some(target) = &osr_target {
            target.executor().validate_entry_profile(&profile_values)?;
        }
        let function_ir = self
            .functions
            .iter()
            .find(|candidate| candidate.id == function)
            .cloned()
            .ok_or_else(|| {
                NativeExecutorError::Host(format!("native function {} is unavailable", function.0))
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
                return Err(NativeExecutorError::Host(
                    "OSR target is not an exact for-loop header in this function".into(),
                ));
            }
        }
        let requested_outputs = u32::try_from(requested_outputs).map_err(|_| {
            NativeExecutorError::Host("requested output count exceeds native ABI".into())
        })?;
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
            workspace,
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
        Ok(NativeInvocation::new(
            state,
            entrypoint,
            argument_refs,
            requested_outputs,
            resume,
            osr_target,
        ))
    }

    fn validate_entry_profile(&self, arguments: &[Value]) -> NativeExecutorResult<()> {
        let Some(profile) = &self.entry_profile else {
            return Ok(());
        };
        let actual = arguments
            .iter()
            .map(runmat_runtime::value_fact::value_fact)
            .collect::<Vec<_>>();
        let actual = crate::RepresentationProfile::from_facts(actual, usize::MAX)
            .map_err(|error| NativeExecutorError::Host(error.into()))?;
        if actual.digest != profile.digest || actual.facts != profile.facts {
            return Err(NativeExecutorError::Host(
                "specialized native entry representation guard failed".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::NativeExecutor;

    fn assert_send<T: Send>() {}

    #[test]
    fn compiled_executor_can_move_from_a_background_compiler_to_its_session_owner() {
        assert_send::<NativeExecutor>();
    }
}
