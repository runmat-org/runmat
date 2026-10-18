use crate::bytecode::program::ExecutionContext;
use crate::bytecode::Bytecode;
use crate::interpreter::vars::InterpreterVars;
use log::debug;
#[cfg(feature = "native-accel")]
use runmat_accelerate::graph::AccelGraph;
#[cfg(feature = "native-accel")]
use runmat_accelerate::FusionPlanRef;
use runmat_gc::{Trace, Tracer};
use runmat_runtime::context::RuntimeInvocationArity;
use runmat_runtime::workspace::bindings::FrameGlobalBindings;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub enum InterpreterOutcome {
    Completed(runmat_runtime::execution::RootedResult<InterpreterCompletion>),
}

/// Final VM frame together with the slots that are semantically assigned.
///
/// The value vector remains dense for efficient indexed execution. Consumers
/// that cross a language boundary must consult `assigned_slots` instead of
/// treating the internal placeholder in an unassigned slot as a program
/// value.
#[derive(Debug)]
pub struct InterpreterCompletion {
    pub values: InterpreterVars,
    pub assigned_slots: HashSet<usize>,
}

impl Trace for InterpreterCompletion {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in self.values.iter() {
            value.trace(tracer);
        }
    }
}

/// Compiler-owned point at which a resumed region completes without executing
/// instructions that belong to its caller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterpreterCompletionBoundary {
    pc: usize,
}

impl InterpreterCompletionBoundary {
    pub const fn before(pc: usize) -> Self {
        Self { pc }
    }

    pub const fn pc(self) -> usize {
        self.pc
    }
}

/// Exact VM-owned state materialized from a native empty-stack MIR boundary.
#[derive(Debug)]
pub struct InterpreterResumeState {
    pub pc: usize,
    pub completion_boundary: Option<InterpreterCompletionBoundary>,
    pub vars: InterpreterVars,
    pub assigned_slots: HashSet<usize>,
    pub supplied_inputs: usize,
    pub requested_outputs: usize,
    pub missing_input_slots: HashSet<usize>,
    pub global_aliases: HashMap<usize, runmat_runtime::workspace::bindings::FrameGlobalBinding>,
    pub persistent_aliases: crate::interpreter::persistent::PersistentAliases,
    pub side_effect_epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveTryHandler {
    pub scope: usize,
    pub catch_pc: usize,
    pub catch_var: Option<usize>,
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "native-accel",
    feature = "wgpu"
))]
#[derive(Debug)]
pub(crate) enum VmFusionPublicationFaultRoute {
    AnySelected,
    SharedRuntime,
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "native-accel",
    feature = "wgpu"
))]
#[derive(Debug)]
pub(crate) enum VmFusionPublicationFault {
    /// Corrupt the pending stack value at the real checked VM StoreVar so the
    /// ordinary dispatch rejects publication and follows its catch path.
    Reject(
        VmFusionPublicationFaultRoute,
        std::rc::Rc<std::cell::Cell<bool>>,
    ),
    Cancel(
        VmFusionPublicationFaultRoute,
        runmat_execution::cancellation::CancellationSource,
        std::rc::Rc<std::cell::Cell<bool>>,
    ),
    Suspend(
        VmFusionPublicationFaultRoute,
        std::rc::Rc<std::cell::Cell<bool>>,
    ),
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "native-accel",
    feature = "wgpu"
))]
impl VmFusionPublicationFault {
    pub(crate) fn route(&self) -> &VmFusionPublicationFaultRoute {
        match self {
            Self::Reject(route, _) | Self::Cancel(route, _, _) | Self::Suspend(route, _) => route,
        }
    }
}

#[derive(Debug)]
pub struct InterpreterState {
    pub bytecode: Bytecode,
    pub stack: Vec<Value>,
    pub vars: InterpreterVars,
    pub pc: usize,
    pub completion_boundary: Option<InterpreterCompletionBoundary>,
    pub completion_assigned_slots: Option<HashSet<usize>>,
    pub context: ExecutionContext,
    pub try_stack: Vec<ActiveTryHandler>,
    pub last_exception: Option<runmat_value::MException>,
    pub imports: Vec<(Vec<String>, bool)>,
    pub global_aliases: FrameGlobalBindings,
    pub persistent_aliases: crate::interpreter::persistent::PersistentAliases,
    pub missing_input_slots: HashSet<usize>,
    pub current_function_name: String,
    pub initial_assigned_var_count: usize,
    #[cfg(feature = "native-accel")]
    pub fusion_plan: Option<FusionPlanRef>,
    #[cfg(feature = "native-accel")]
    pub fusion_accel_graph: Option<AccelGraph>,
    #[cfg(feature = "native-accel")]
    pub(crate) fusion_installation: Option<crate::bytecode::program::VmFusionPlanInstallation>,
    #[cfg(all(
        test,
        not(target_arch = "wasm32"),
        feature = "native-accel",
        feature = "wgpu"
    ))]
    pub(crate) fusion_publication_fault: Option<VmFusionPublicationFault>,
}

impl InterpreterState {
    pub fn new(
        bytecode: Bytecode,
        initial_vars: &mut [Value],
        current_function_name: Option<&str>,
        invocation_arity: RuntimeInvocationArity,
    ) -> Self {
        Self::new_in_context(
            bytecode,
            initial_vars,
            current_function_name,
            invocation_arity,
            runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            )),
        )
    }

    pub fn new_in_context(
        bytecode: Bytecode,
        initial_vars: &mut [Value],
        current_function_name: Option<&str>,
        invocation_arity: RuntimeInvocationArity,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Self {
        Self::from_owned_in_context(
            bytecode,
            initial_vars.to_vec(),
            current_function_name,
            invocation_arity,
            runtime,
        )
    }

    pub fn from_owned_in_context(
        bytecode: Bytecode,
        initial_vars: Vec<Value>,
        current_function_name: Option<&str>,
        invocation_arity: RuntimeInvocationArity,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Self {
        let initial_assigned_var_count = initial_vars.len();
        let mut vars = InterpreterVars::from_owned(initial_vars);
        if vars.len() < bytecode.var_count {
            vars.resize_owned(bytecode.var_count, Value::Num(0.0));
        }
        Self::from_prepared_vars_in_context(
            bytecode,
            vars,
            initial_assigned_var_count,
            current_function_name,
            invocation_arity,
            runtime,
        )
    }

    pub fn from_vars_in_context(
        bytecode: Bytecode,
        vars: InterpreterVars,
        initial_assigned_var_count: usize,
        current_function_name: Option<&str>,
        invocation_arity: RuntimeInvocationArity,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Result<Self, runmat_runtime::RuntimeError> {
        if vars.len() < bytecode.var_count {
            return Err(runmat_runtime::execution::ExecutionServiceError::Infrastructure(
                "RunMat:Interpreter:RetainedFrameExtent: retained interpreter frame extent was not prepared before execution"
                    .to_owned(),
            )
            .into_runtime_error());
        }
        Ok(Self::from_prepared_vars_in_context(
            bytecode,
            vars,
            initial_assigned_var_count,
            current_function_name,
            invocation_arity,
            runtime,
        ))
    }

    fn from_prepared_vars_in_context(
        bytecode: Bytecode,
        vars: InterpreterVars,
        initial_assigned_var_count: usize,
        current_function_name: Option<&str>,
        invocation_arity: RuntimeInvocationArity,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Self {
        if bytecode.async_metadata.mir_spawn_site_count > 0
            || bytecode.async_metadata.mir_await_site_count > 0
        {
            debug!(
                "async semantics: compiled bytecode carries {} MIR spawn site(s) and {} MIR await site(s); runtime model={} with explicit spawn/await bytecode boundaries",
                bytecode.async_metadata.mir_spawn_site_count,
                bytecode.async_metadata.mir_await_site_count,
                bytecode.async_metadata.runtime_model.as_str()
            );
        }
        #[cfg(feature = "native-accel")]
        let (fusion_plan, fusion_accel_graph, fusion_installation) =
            bytecode.prepare_vm_fusion_plan_with_installation(&runtime);
        let graph_program = runtime.selected_graph_program().cloned();
        Self {
            stack: Vec::new(),
            context: ExecutionContext {
                call_stack: Vec::new(),
                locals: Vec::new(),
                instruction_pointer: 0,
                runtime: runtime.with_invocation_arity(invocation_arity),
                lexical_environment: None,
                lexical_binding_slots: Vec::new(),
                inherited_storage_slots: Vec::new(),
                graph_program,
                process_driver: None,
                dynamic_eval_admission: None,
            },
            try_stack: Vec::new(),
            last_exception: None,
            imports: Vec::new(),
            global_aliases: FrameGlobalBindings::default(),
            persistent_aliases: HashMap::new(),
            missing_input_slots: HashSet::new(),
            vars,
            pc: 0,
            completion_boundary: None,
            completion_assigned_slots: None,
            initial_assigned_var_count,
            current_function_name: current_function_name
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<main>".to_string()),
            #[cfg(feature = "native-accel")]
            fusion_plan,
            #[cfg(feature = "native-accel")]
            fusion_accel_graph,
            #[cfg(feature = "native-accel")]
            fusion_installation,
            #[cfg(all(
                test,
                not(target_arch = "wasm32"),
                feature = "native-accel",
                feature = "wgpu"
            ))]
            fusion_publication_fault: None,
            bytecode,
        }
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use super::InterpreterState;
    use crate::bytecode::{Bytecode, FusionInstructionKind, FusionInstructionWindow};
    use runmat_accelerate::graph::{AccelNodeLabel, InstrSpan, PrimitiveOp};
    use runmat_runtime::context::RuntimeInvocationArity;

    #[test]
    fn runtime_materialized_graph_is_retained_for_fusion_execution() {
        let mut bytecode = Bytecode::empty();
        bytecode.instructions = vec![
            crate::Instr::LoadVar(0),
            crate::Instr::LoadVar(1),
            crate::Instr::Add,
        ];
        bytecode.var_types = vec![
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
        ];
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 2, end: 2 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let mut initial_vars = Vec::new();
        let state = InterpreterState::new(
            bytecode,
            &mut initial_vars,
            Some("<main>"),
            RuntimeInvocationArity::Script,
        );
        assert!(
            state.fusion_plan.is_some(),
            "expected runtime fusion plan when semantic windows exist"
        );
        assert!(
            state.fusion_accel_graph.is_some(),
            "expected runtime accel graph to be retained for fusion execution"
        );
    }

    #[test]
    fn runtime_state_ignores_stale_compile_graph_metadata() {
        let mut bytecode = Bytecode::empty();
        bytecode.instructions = vec![
            crate::Instr::LoadVar(0),
            crate::Instr::LoadVar(1),
            crate::Instr::Add,
        ];
        bytecode.var_types = vec![
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
        ];
        let stale_graph = crate::accel::graph::build_accel_graph(
            &[
                crate::Instr::LoadVar(0),
                crate::Instr::LoadVar(1),
                crate::Instr::Mul,
            ],
            &bytecode.var_types,
        );
        bytecode.accel_graph = Some(stale_graph);
        bytecode.fusion_metadata.mir_fusion_candidate_group_count = 1;
        bytecode.fusion_metadata.instruction_windows = vec![FusionInstructionWindow {
            span: InstrSpan { start: 2, end: 2 },
            kind: FusionInstructionKind::Elementwise,
        }];

        let mut initial_vars = Vec::new();
        let state = InterpreterState::new(
            bytecode,
            &mut initial_vars,
            Some("<main>"),
            RuntimeInvocationArity::Script,
        );
        let graph = state
            .fusion_accel_graph
            .as_ref()
            .expect("expected runtime accel graph to be retained");
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| matches!(node.label, AccelNodeLabel::Primitive(PrimitiveOp::Add))),
            "runtime retained graph should reflect active bytecode instructions"
        );
        assert!(
            !graph
                .nodes
                .iter()
                .any(|node| matches!(node.label, AccelNodeLabel::Primitive(PrimitiveOp::Mul))),
            "stale compile graph metadata should not be retained in runtime state"
        );
    }
}
