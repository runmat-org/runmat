use super::*;
use crate::execute::workspace::NativeWorkspaceState;
use crate::memory::ScopedValueRoots;
use runmat_runtime::call::function_abi::{prepare_function_inputs, FunctionInputSpec};

impl HostState {
    pub fn new(input: HostStateInput) -> NativeExecutorResult<(Self, Vec<NativeValueRef>)> {
        let HostStateInput {
            function,
            arguments,
            requested_outputs,
            runtime,
            program_capture,
            functions,
            captures,
            deoptimization,
            interpreter_resume_points,
            coverage_sites,
            osr_point,
            optimized_regions,
            workspace,
        } = input;
        // Workspace entry is a script frame; ordinary native invocation owns
        // the exact supplied/requested function arity. Project it before any
        // callsite can suspend or reach the runtime builtin dispatcher.
        let invocation_arity = if workspace.is_some() {
            runmat_runtime::context::RuntimeInvocationArity::Script
        } else {
            runmat_runtime::context::RuntimeInvocationArity::function(
                arguments.len(),
                requested_outputs,
            )
        };
        let source_id = runtime.admitted_program_source_id(function.source)?;
        let entry_source = function
            .blocks
            .iter()
            .find(|block| block.id == function.entry)
            .map(|block| {
                block
                    .instructions
                    .first()
                    .map_or(&block.terminator.source, |instruction| &instruction.source)
            });
        let frame_span = entry_source
            .filter(|source| source.source == function.source)
            .and_then(|source| {
                let start = usize::try_from(source.span.start).ok()?;
                let end = usize::try_from(source.span.end).ok()?;
                (end >= start).then_some((start, end))
            });
        let runtime = runtime
            .with_invocation_arity(invocation_arity)
            .with_current_program_source(function.source)?
            .with_debug_frame(runmat_runtime::debug_context::DebugFrame {
                function: function.name.clone(),
                source_id,
                span: frame_span,
            });
        let inherited_catch = runtime.active_catch_exception_shared();
        let construction_values = arguments
            .iter()
            .cloned()
            .chain(captures.iter().map(|capture| capture.value.clone()))
            .collect();
        let _construction_roots =
            ScopedValueRoots::register(construction_values, "native_invocation_construction")?;
        let input_specs = function
            .argument_validations
            .iter()
            .map(|validation| {
                let input_index = function
                    .abi
                    .fixed_inputs
                    .iter()
                    .position(|local| *local == validation.input)
                    .ok_or_else(|| {
                        NativeExecutorError::Host(
                            "native argument validation does not name a fixed input".into(),
                        )
                    })?;
                Ok(FunctionInputSpec {
                    input_index,
                    size: validation.size.as_ref(),
                    class_name: validation.class_name.as_deref(),
                    validators: &validation.validators,
                    default_value: validation.default_value.as_ref(),
                })
            })
            .collect::<NativeExecutorResult<Vec<_>>>()?;
        let supplied_inputs = arguments.len();
        let prepared = prepare_function_inputs(
            &function.name,
            &arguments,
            function.abi.fixed_inputs.len(),
            function.abi.varargin.is_some(),
            &input_specs,
        )?;
        let missing_input_locals = function
            .abi
            .fixed_inputs
            .iter()
            .zip(&prepared.fixed)
            .filter_map(|(local, value)| value.is_none().then_some(*local))
            .collect();
        let global_bindings = workspace
            .as_ref()
            .map(|input| input.frame_global_bindings(&function))
            .transpose()?
            .unwrap_or_default();
        let canonical_workspace = runtime.service_ports().workspace().cloned();
        let workspace = if let Some(input) = workspace {
            NativeWorkspaceState::new(
                &function,
                input,
                global_bindings.clone(),
                canonical_workspace,
                Some(runtime.session_variable_authority()),
            )?
        } else {
            NativeWorkspaceState::function_frame(
                &function,
                global_bindings.clone(),
                canonical_workspace,
                Some(runtime.session_variable_authority()),
            )
        };
        let ports = runtime
            .service_ports()
            .clone()
            .with_workspace(workspace.service());
        let runtime = runtime.with_service_ports(ports);
        let mut arena = ValueArena::new()?;
        let argument_refs = arguments
            .into_iter()
            .map(|value| arena.insert(value))
            .collect::<Vec<_>>();
        let mut locals = vec![NativeValueRef::NULL; function.local_count()];
        let capture_locals = function
            .locals
            .iter()
            .filter(|local| local.kind == NativeLocalKind::Capture)
            .collect::<Vec<_>>();
        if capture_locals.len() != captures.len() {
            return Err(NativeExecutorError::Host(format!(
                "native lexical entry expected {} captures but received {}",
                capture_locals.len(),
                captures.len()
            )));
        }
        for local in capture_locals {
            let binding = local.binding.ok_or_else(|| {
                NativeExecutorError::Host("native capture local has no semantic binding".into())
            })?;
            let capture = captures
                .iter()
                .find(|capture| capture.binding == binding)
                .ok_or_else(|| {
                    NativeExecutorError::Host("native lexical capture binding is missing".into())
                })?;
            locals[local.id.0 as usize] = arena.insert(capture.value.clone());
        }
        for (local, value) in function.abi.fixed_inputs.iter().zip(&prepared.fixed) {
            let slot = locals.get_mut(local.0 as usize).ok_or_else(|| {
                NativeExecutorError::Host("function ABI input local is out of bounds".into())
            })?;
            if let Some(value) = value {
                *slot = arena.insert(value.clone());
            }
        }
        if let (Some(local), Some(varargin)) = (function.abi.varargin, prepared.varargin) {
            locals[local.0 as usize] = arena.insert(runmat_value::Value::Cell(varargin));
        }
        if let Some(local) = function.abi.varargout {
            let empty = runmat_value::CellArray::new(Vec::new(), 1, 0)
                .map_err(|error| NativeExecutorError::Host(format!("varargout: {error}")))?;
            locals[local.0 as usize] = arena.insert(runmat_value::Value::Cell(empty));
        }
        if let Some(local) = function.abi.implicit_nargin {
            locals[local.0 as usize] =
                arena.insert(runmat_value::Value::Num(prepared.nargin as f64));
        }
        if let Some(local) = function.abi.implicit_nargout {
            locals[local.0 as usize] =
                arena.insert(runmat_value::Value::Num(requested_outputs as f64));
        }
        for (slot, local) in locals.iter_mut().enumerate() {
            if let Some(value) = workspace.initial_value(slot) {
                *local = arena.insert(value);
            }
            let reference = *local;
            if !reference.is_null() {
                workspace.initialize_local(slot, arena.get(reference)?.clone())?;
            }
        }
        let roots = locals
            .iter()
            .enumerate()
            .map(|(slot, value)| NativeRoot {
                value: *value,
                kind: NativeRootKind::LOCAL,
                slot: slot as u32,
            })
            .collect::<Vec<_>>();
        let source = function.source.0;
        let mut state = Self {
            function,
            functions,
            runtime,
            inherited_catch,
            arena,
            locals,
            values: BTreeMap::new(),
            roots,
            last_error: None,
            host_failure: None,
            current_source: runmat_runtime::native::NativeSourceLocation {
                source,
                ..runmat_runtime::native::NativeSourceLocation::default()
            },
            active_for_loops: BTreeMap::new(),
            loop_backedges: BTreeMap::new(),
            osr_point,
            osr_entry: None,
            pending_place_mutation: None,
            pending_assign: None,
            pending_place_prepare: None,
            pending_plain_multi: None,
            program_capture,
            pending_await: None,
            pending_call: None,
            completed_call: None,
            sequence_assignment_register: None,
            pending_sequence_rvalue: None,
            pending_sequence_assign: None,
            prepared_multi_assignment: None,
            pending_multi_preparation: None,
            captured_sequences: BTreeMap::new(),
            contextual_index_extents: Vec::new(),
            subscript_end_receivers: Vec::new(),
            contextual_regions: contextual_regions::ContextualRegionState::default(),
            current_request: None,
            resume_target: None,
            current_block: None,
            global_bindings,
            global_seen_revision: BTreeMap::new(),
            persistent_bindings: BTreeMap::new(),
            active_exception_handlers: Vec::new(),
            catch_exceptions: BTreeMap::new(),
            next_await_continuation: 1,
            deoptimization,
            retired_guards: BTreeSet::new(),
            pending_deoptimization: None,
            interpreter_resume_points,
            coverage_sites,
            supplied_inputs,
            requested_outputs,
            missing_input_locals,
            optimized_regions: optimized_regions
                .iter()
                .cloned()
                .map(|plan| (plan.entry, plan))
                .collect(),
            region_observation: None,
            skipped_optimized_sites: BTreeSet::new(),
            vectorized_regions: 0,
            workspace: Some(workspace),
            last_expression: None,
        };
        state.enter_block(state.function.entry)?;
        Ok((state, argument_refs))
    }
}
