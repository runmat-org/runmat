use runmat_gc::{Trace, Tracer};
use runmat_runtime::context::RuntimeContext;
use runmat_types::ProgramFunctionId;
use runmat_value::Value;
use std::{collections::BTreeMap, sync::Arc};

use crate::deopt::DeoptimizationPolicy;
use crate::osr::OsrTarget;
use crate::{NativeExecutable, NativeExecutorError, NativeExecutorResult};

use super::invocation::{NativeInvocation, NativeInvocationStep};
use super::state::{HostState, HostStateInput, NativeCaptureInput, NativeFunctionInputs};

mod capture_arguments;

pub struct NativeExecutor {
    functions: Arc<Vec<runmat_native_codegen::NativeFunction>>,
    class_declarations: Arc<[runmat_native_codegen::NativeClassDeclaration]>,
    program_revision: runmat_execution::ProgramRevision,
    compiled: NativeExecutable,
    interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
    pub(super) regions: Vec<runmat_types::RegionContract>,
    pub(super) compile_duration_ns: u64,
    entry_profile: Option<EntryRepresentationProfile>,
    optimized_regions: Arc<Vec<crate::region::OptimizedRegionPlan>>,
    required_runtime_services:
        std::collections::BTreeSet<runmat_execution::host::NativeDeploymentServiceKind>,
    selected_source_evaluation: Option<runmat_execution::host::VerifiedNativeServiceIssuer>,
}

struct EntryRepresentationProfile {
    function: ProgramFunctionId,
    profile: crate::RepresentationProfile,
}

#[derive(Default)]
pub struct NativeExecutorOptions {
    pub interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    pub coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
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

struct NativeRequestParts {
    function: ProgramFunctionId,
    inputs: NativeFunctionInputs,
    captures: Vec<NativeCaptureInput>,
    requested_outputs: usize,
    runtime: RuntimeContext,
    deoptimization: DeoptimizationPolicy,
    osr_target: Option<OsrTarget>,
    workspace: Option<super::workspace::NativeWorkspaceInput>,
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

/// A terminal native result that retains managed values after its invocation
/// arena is released, until a downstream owner publishes the result.
pub type RootedNativeExecution = runmat_runtime::execution::RootedResult<NativeExecution>;

impl Trace for NativeExecution {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in &self.outputs {
            value.trace(tracer);
        }
        for capture in &self.captures {
            capture.value.trace(tracer);
        }
        if let Some(workspace) = &self.workspace {
            for (_, value) in workspace.iter() {
                value.trace(tracer);
            }
        }
        if let Some(expression) = &self.expression {
            expression.trace(tracer);
        }
    }
}

impl NativeExecutor {
    /// Selected callable entry keeps captures in their immutable callable
    /// generation and moves only the explicit suffix into function locals.
    pub async fn invoke_rooted_function_arguments_async(
        &self,
        function: ProgramFunctionId,
        arguments: runmat_runtime::call::RootedFunctionArguments,
        requested_outputs: usize,
        runtime: RuntimeContext,
        lexical_environment: Option<runmat_value::LexicalEnvironment>,
    ) -> NativeExecutorResult<RootedNativeExecution> {
        let invocation = self
            .begin_rooted_function_arguments_async(
                function,
                arguments,
                requested_outputs,
                runtime,
                lexical_environment,
            )
            .await?;
        Self::drive_invocation(invocation).await
    }

    /// Prepare a selected rooted callable invocation without consuming its
    /// suspension or deoptimization exits. Core uses this entry to apply the
    /// same interpreter-resume policy as ordinary native execution.
    pub async fn begin_rooted_function_arguments_async(
        &self,
        function: ProgramFunctionId,
        arguments: runmat_runtime::call::RootedFunctionArguments,
        requested_outputs: usize,
        runtime: RuntimeContext,
        lexical_environment: Option<runmat_value::LexicalEnvironment>,
    ) -> NativeExecutorResult<NativeInvocation> {
        self.require_runtime_services(&runtime)?;
        let retained_captures = self.retained_capture_arguments(function, &arguments)?;
        self.validate_entry_profile(function, arguments.supplied().iter())?;
        let function_ir = self
            .functions
            .iter()
            .find(|body| body.id == function)
            .ok_or_else(|| {
                NativeExecutorError::Host(format!("native function {} is unavailable", function.0))
            })?;
        arguments
            .with_owned_supplied_async(|_callable, supplied| async move {
                let _callable = _callable;
                let specs = super::state::function_input_specs(function_ir)?;
                let prepared = runmat_runtime::call::function_abi::prepare_function_inputs_owned(
                    &function_ir.name,
                    supplied,
                    function_ir.abi.fixed_inputs.len(),
                    function_ir.abi.varargin.is_some(),
                    &specs,
                )
                .await?;
                let request = NativeRequestParts {
                    function,
                    captures: retained_captures,
                    inputs: NativeFunctionInputs::Prepared(prepared),
                    requested_outputs,
                    runtime,
                    deoptimization: DeoptimizationPolicy::default(),
                    osr_target: None,
                    workspace: None,
                };
                self.begin_request_with_environment_parts(request, lexical_environment)
            })
            .await?
    }

    /// Bind verified Native IR to an executable with the same exact function set.
    pub fn bind(
        assembly: runmat_native_codegen::NativeAssembly,
        executable: NativeExecutable,
        options: NativeExecutorOptions,
    ) -> NativeExecutorResult<Self> {
        Self::bind_with_selected_service(assembly, executable, options, None)
    }

    /// Selected entry requires the already verified target-pack product, never
    /// a decoded product declaration or a service identity copied by a caller.
    pub fn bind_selected(
        assembly: runmat_native_codegen::NativeAssembly,
        executable: NativeExecutable,
        options: NativeExecutorOptions,
        selected: &runmat_native_codegen::aot::SelectedNativeProduct<'_>,
    ) -> NativeExecutorResult<Self> {
        selected.validate_assembly(&assembly)?;
        let issuer = selected.source_evaluation_issuer()?;
        Self::bind_with_selected_service(assembly, executable, options, issuer)
    }

    /// Bound only by the generated standalone linked-image entry.
    pub fn bind_linked_selected(
        assembly: runmat_native_codegen::NativeAssembly,
        executable: NativeExecutable,
        options: NativeExecutorOptions,
        linked: &runmat_native_codegen::aot::LinkedNativeProduct,
        source_evaluation: Option<runmat_execution::host::VerifiedNativeServiceIssuer>,
    ) -> NativeExecutorResult<Self> {
        linked.validate_assembly(&assembly)?;
        if source_evaluation.as_ref().map(|issuer| issuer.product())
            != linked
                .binding()
                .required_services
                .contains(&runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation)
                .then(|| linked.binding().fingerprint())
                .transpose()
                .map_err(|error| NativeExecutorError::Executable(error.to_string()))?
        {
            return Err(NativeExecutorError::Executable(
                "linked source service differs from selected product".into(),
            ));
        }
        Self::bind_with_selected_service(assembly, executable, options, source_evaluation)
    }

    fn bind_with_selected_service(
        assembly: runmat_native_codegen::NativeAssembly,
        executable: NativeExecutable,
        options: NativeExecutorOptions,
        selected_source_evaluation: Option<runmat_execution::host::VerifiedNativeServiceIssuer>,
    ) -> NativeExecutorResult<Self> {
        assembly.verify()?;
        assembly.target.require_host_executable()?;
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
        let required_runtime_services = assembly.required_runtime_services();
        let profiled_function = assembly.selected_entrypoint;
        let optimized_regions = if options.entry_profile.is_some() {
            crate::region::derive_plans(&assembly.functions, &regions)
        } else {
            Vec::new()
        };
        Ok(Self {
            functions: Arc::new(assembly.functions),
            class_declarations: Arc::from(assembly.class_declarations),
            program_revision: assembly.program,
            compiled: executable,
            interpreter_resume_points: options.interpreter_resume_points,
            coverage_sites: options.coverage_sites,
            regions,
            required_runtime_services,
            selected_source_evaluation,
            compile_duration_ns: options.compile_duration_ns,
            entry_profile: options
                .entry_profile
                .map(|profile| EntryRepresentationProfile {
                    function: profiled_function,
                    profile,
                }),
            optimized_regions: Arc::new(optimized_regions),
        })
    }

    /// The exact class declaration projection admitted with this executor.
    pub fn class_declarations(&self) -> &[runmat_native_codegen::NativeClassDeclaration] {
        &self.class_declarations
    }

    pub fn function_metadata(
        &self,
        function: ProgramFunctionId,
    ) -> Option<&runmat_native_codegen::NativeFunction> {
        self.functions
            .iter()
            .find(|candidate| candidate.id == function)
    }

    pub fn program_revision(&self) -> &runmat_execution::ProgramRevision {
        &self.program_revision
    }

    pub fn retained_code_bytes(&self) -> u64 {
        self.compiled.retained_code_bytes()
    }

    pub fn entry_profile(&self) -> Option<&crate::RepresentationProfile> {
        self.entry_profile.as_ref().map(|entry| &entry.profile)
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
    ) -> NativeExecutorResult<RootedNativeExecution> {
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
    ) -> NativeExecutorResult<RootedNativeExecution> {
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
    ) -> NativeExecutorResult<RootedNativeExecution> {
        self.invoke_request_async_in_environment(
            NativeInvocationRequest {
                function,
                captures,
                arguments,
                requested_outputs,
                runtime,
                deoptimization: DeoptimizationPolicy::default(),
                osr_target: None,
                workspace: None,
            },
            None,
        )
        .await
    }

    pub async fn invoke_request_async_in_environment(
        &self,
        request: NativeInvocationRequest,
        lexical_environment: Option<runmat_value::LexicalEnvironment>,
    ) -> NativeExecutorResult<RootedNativeExecution> {
        let invocation = self.begin_request_with_environment(request, lexical_environment)?;
        Self::drive_invocation(invocation).await
    }

    async fn drive_invocation(
        mut invocation: NativeInvocation,
    ) -> NativeExecutorResult<RootedNativeExecution> {
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
    ) -> NativeExecutorResult<RootedNativeExecution> {
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
        self.begin_request_with_environment(request, None)
    }

    /// Carry the exact admitted activation through native construction.
    /// Ordinary requests remain on the environment-free path.
    pub fn begin_request_with_environment(
        &self,
        request: NativeInvocationRequest,
        lexical_environment: Option<runmat_value::LexicalEnvironment>,
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
        self.begin_request_with_environment_parts(
            NativeRequestParts {
                function,
                captures: captures
                    .into_iter()
                    .map(NativeCaptureInput::Owned)
                    .collect(),
                inputs: NativeFunctionInputs::Legacy(arguments),
                requested_outputs,
                runtime,
                deoptimization,
                osr_target,
                workspace,
            },
            lexical_environment,
        )
    }

    fn begin_request_with_environment_parts(
        &self,
        request: NativeRequestParts,
        lexical_environment: Option<runmat_value::LexicalEnvironment>,
    ) -> NativeExecutorResult<NativeInvocation> {
        // A pre-cancelled invocation must not begin constructing a frame or
        // report an admission failure before the execution cancellation exit.
        if request.runtime.is_cancelled() {
            return Err(NativeExecutorError::Cancelled);
        }
        let _admission_roots = lexical_environment
            .as_ref()
            .map(runmat_runtime::call::lexical::root_environment)
            .transpose()?;
        let NativeRequestParts {
            function,
            captures,
            inputs,
            requested_outputs,
            runtime,
            deoptimization,
            osr_target,
            workspace,
        } = request;
        self.require_runtime_services(&runtime)?;
        let profile_values = || {
            let arguments = match &inputs {
                NativeFunctionInputs::Legacy(arguments) => arguments.as_slice(),
                NativeFunctionInputs::Prepared(_) => &[],
            };
            arguments.iter().chain(
                workspace
                    .as_ref()
                    .into_iter()
                    .flat_map(|workspace| workspace.profile_values()),
            )
        };
        if matches!(&inputs, NativeFunctionInputs::Legacy(_)) {
            self.validate_entry_profile(function, profile_values())?;
        }
        if let Some(target) = &osr_target {
            if matches!(&inputs, NativeFunctionInputs::Legacy(_)) {
                target
                    .executor()
                    .validate_entry_profile(function, profile_values())?;
            }
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
            inputs,
            requested_outputs: requested_outputs as usize,
            runtime,
            functions: Arc::clone(&self.functions),
            captures,
            lexical_environment,
            deoptimization,
            interpreter_resume_points: self.interpreter_resume_points.clone(),
            coverage_sites: self.coverage_sites.clone(),
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

    fn require_runtime_services(&self, runtime: &RuntimeContext) -> NativeExecutorResult<()> {
        if self
            .required_runtime_services
            .contains(&runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation)
        {
            if let Some(issuer) = &self.selected_source_evaluation {
                runtime
                    .service_ports()
                    .require_selected_source_evaluation(issuer)?;
            } else {
                runtime
                    .service_ports()
                    .require_source_evaluation("execute native source evaluation")
                    .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
            }
        }
        Ok(())
    }

    fn validate_entry_profile<'a>(
        &self,
        function: ProgramFunctionId,
        arguments: impl Iterator<Item = &'a Value>,
    ) -> NativeExecutorResult<()> {
        let Some(entry) = &self.entry_profile else {
            return Ok(());
        };
        // A compiled product may contain local functions in addition to its
        // selected entrypoint. Only the selected function was specialized for
        // this invocation profile; its nested callees have independent inputs.
        if function != entry.function {
            return Ok(());
        }
        let actual = arguments
            .map(runmat_runtime::value_fact::value_fact)
            .collect::<Vec<_>>();
        let actual = crate::RepresentationProfile::from_facts(actual, usize::MAX)
            .map_err(|error| NativeExecutorError::Host(error.into()))?;
        if actual.digest != entry.profile.digest || actual.facts != entry.profile.facts {
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
