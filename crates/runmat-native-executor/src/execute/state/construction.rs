use super::*;
use crate::execute::workspace::NativeWorkspaceState;
use crate::memory::ScopedValueRoots;
use runmat_runtime::call::function_abi::{prepare_function_inputs, FunctionInputSpec};

// This is a move-only handoff between context/input admission and local arena
// installation. In particular, construction_roots must survive the handoff;
// prepared legacy inputs cannot lose their roots while locals are seeded.
struct PreparedConstruction {
    function: NativeFunction,
    requested_outputs: usize,
    runtime: RuntimeContext,
    program_capture: Option<Vec<u8>>,
    functions: Arc<Vec<NativeFunction>>,
    captures: Vec<NativeCaptureInput>,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
    lexical_roots: Vec<runmat_gc::ExplicitRoot>,
    deoptimization: DeoptimizationPolicy,
    interpreter_resume_points: BTreeMap<runmat_types::ProgramPointId, u64>,
    coverage_sites: BTreeMap<runmat_native_codegen::NativeMirSite, Vec<u64>>,
    osr_point: Option<runmat_types::ProgramPointId>,
    optimized_regions: Arc<Vec<crate::region::OptimizedRegionPlan>>,
    contextual_regions: expression_regions::ExpressionRegionState,
    inherited_catch: Option<std::rc::Rc<runmat_value::MException>>,
    arguments: Option<Vec<runmat_value::Value>>,
    prepared: runmat_runtime::call::function_abi::PreparedFunctionInputs,
    construction_roots: Option<ScopedValueRoots>,
    supplied_inputs: usize,
    missing_input_locals: Vec<runmat_native_codegen::NativeLocalId>,
    global_bindings: runmat_runtime::workspace::bindings::FrameGlobalBindings,
    workspace: NativeWorkspaceState,
}

impl HostState {
    pub fn new(input: HostStateInput) -> NativeExecutorResult<(Self, Vec<NativeValueRef>)> {
        Self::install_locals(Self::prepare_context(input)?)
    }

    // No arena or pending-site state is constructed while source admission,
    // lexical ownership, and ABI preparation can call into other owners.
    #[inline(never)]
    fn prepare_context(input: HostStateInput) -> NativeExecutorResult<PreparedConstruction> {
        let HostStateInput {
            function,
            inputs,
            requested_outputs,
            runtime,
            program_capture,
            functions,
            captures,
            lexical_environment,
            deoptimization,
            interpreter_resume_points,
            coverage_sites,
            osr_point,
            optimized_regions,
            workspace,
        } = input;
        let (lexical_environment, lexical_roots) = if function.creates_lexical_environment {
            let (environment, roots) =
                runmat_runtime::call::lexical::allocate_environment(lexical_environment)?;
            (Some(environment), roots)
        } else {
            let roots = lexical_environment
                .as_ref()
                .map(runmat_runtime::call::lexical::root_environment)
                .transpose()?
                .unwrap_or_default();
            (lexical_environment, roots)
        };
        // Workspace entry is a script frame; ordinary native invocation owns
        // the exact supplied/requested function arity. Project it before any
        // callsite can suspend or reach the runtime builtin dispatcher.
        let invocation_arity = if workspace.is_some() {
            runmat_runtime::context::RuntimeInvocationArity::Script
        } else {
            runmat_runtime::context::RuntimeInvocationArity::function(
                inputs.input_count(),
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
        // Admit every exact-capacity continuation buffer before argument
        // preparation, workspace attachment, or lexical publication can run.
        let contextual_regions = expression_regions::ExpressionRegionState::prepare(
            &runtime,
            function.expression_continuation_geometry,
        )?;
        let inherited_catch = runtime.active_catch_exception_shared();
        // Compatibility entry owns a second Value forest for the direct ABI.
        // Selected retained entry arrives with one already validated, move-only
        // prepared input set and retains its original roots outside this frame.
        let (arguments, prepared, construction_roots) =
            prepare_inputs(&function, inputs, &captures)?;
        let supplied_inputs = prepared.nargin;
        let missing_input_locals = function
            .abi
            .fixed_inputs
            .iter()
            .zip(&prepared.fixed)
            .filter_map(|(local, value)| value.is_none().then_some(*local))
            .collect();
        let global_bindings = workspace
            .as_ref()
            .map(|input| input.frame_global_bindings(&function, &runtime))
            .transpose()?
            .unwrap_or_default();
        let canonical_workspace = runtime.service_ports().workspace().cloned();
        let workspace = if let Some(input) = workspace {
            NativeWorkspaceState::new_for_context(
                &function,
                input,
                global_bindings.clone(),
                canonical_workspace,
                Some(runtime.session_variable_authority()),
                &runtime,
            )?
        } else {
            NativeWorkspaceState::function_frame_for_context(
                &function,
                global_bindings.clone(),
                canonical_workspace,
                Some(runtime.session_variable_authority()),
                &runtime,
            )?
        };
        workspace.attach_lexical_bindings(&function, lexical_environment.as_ref())?;
        let ports = runtime
            .service_ports()
            .clone()
            .with_workspace(workspace.service());
        let runtime = runtime.with_service_ports(ports);
        Ok(PreparedConstruction {
            function,
            requested_outputs,
            runtime,
            program_capture,
            functions,
            captures,
            lexical_environment,
            lexical_roots,
            deoptimization,
            interpreter_resume_points,
            coverage_sites,
            osr_point,
            optimized_regions,
            contextual_regions,
            inherited_catch,
            arguments,
            prepared,
            construction_roots,
            supplied_inputs,
            missing_input_locals,
            global_bindings,
            workspace,
        })
    }

    #[inline(never)]
    fn install_locals(
        preparation: PreparedConstruction,
    ) -> NativeExecutorResult<(Self, Vec<NativeValueRef>)> {
        let PreparedConstruction {
            function,
            requested_outputs,
            runtime,
            program_capture,
            functions,
            captures,
            lexical_environment,
            lexical_roots,
            deoptimization,
            interpreter_resume_points,
            coverage_sites,
            osr_point,
            optimized_regions,
            contextual_regions,
            inherited_catch,
            arguments,
            prepared,
            construction_roots: _construction_roots,
            supplied_inputs,
            missing_input_locals,
            global_bindings,
            workspace,
        } = preparation;
        let mut arena = ValueArena::new_in(&runtime)?;
        let selected_inputs = arguments.is_none();
        let mut argument_refs = arguments
            .unwrap_or_default()
            .into_iter()
            .map(|value| arena.insert(value, Some(&runtime)))
            .collect::<NativeExecutorResult<Vec<_>>>()?;
        let mut locals = vec![NativeValueRef::NULL; function.local_count()];
        let capture_locals = function
            .closure_captures
            .iter()
            .filter(|capture| {
                matches!(
                    capture.disposition,
                    runmat_native_codegen::NativeClosureCaptureDisposition::ByValue
                        | runmat_native_codegen::NativeClosureCaptureDisposition::TransitSnapshot
                )
            })
            .collect::<Vec<_>>();
        if capture_locals.len() != captures.len() {
            return Err(NativeExecutorError::Host(format!(
                "native lexical entry expected {} captures but received {}",
                capture_locals.len(),
                captures.len()
            )));
        }
        for capture in &captures {
            let (local, value) = match capture {
                NativeCaptureInput::Owned(capture) => {
                    let local = capture_locals
                        .iter()
                        .find(|layout| layout.binding == capture.binding)
                        .ok_or_else(|| {
                            NativeExecutorError::Host(
                                "native lexical capture binding is missing".into(),
                            )
                        })?;
                    (
                        local.local,
                        arena.insert(capture.value.clone(), Some(&runtime))?,
                    )
                }
                NativeCaptureInput::Retained {
                    binding,
                    local,
                    value,
                } => {
                    let matches_layout = function.closure_captures.iter().any(|layout| {
                        layout.binding == *binding
                            && layout.local == *local
                            && matches!(
                                layout.disposition,
                                runmat_native_codegen::NativeClosureCaptureDisposition::ByValue
                                    | runmat_native_codegen::NativeClosureCaptureDisposition::TransitSnapshot
                            )
                    });
                    if !matches_layout {
                        return Err(NativeExecutorError::Host(
                            "retained capture disagrees with the verified Native layout".into(),
                        ));
                    }
                    (
                        *local,
                        arena.insert_capture_lease(value.clone(), Some(&runtime))?,
                    )
                }
            };
            let slot = locals.get_mut(local.0 as usize).ok_or_else(|| {
                NativeExecutorError::Host("native capture local is out of bounds".into())
            })?;
            if !slot.is_null() {
                return Err(NativeExecutorError::Host(
                    "native capture local is initialized more than once".into(),
                ));
            }
            *slot = value;
        }
        if let Some(receiver) = runtime.captured_constructor_receiver().filter(|_| {
            function
                .class_method_owner
                .as_ref()
                .is_some_and(|owner| owner.role == runmat_types::ClassMethodRole::Constructor)
        }) {
            let local = function.abi.fixed_outputs.first().ok_or_else(|| {
                NativeExecutorError::from(
                    runmat_runtime::build_runtime_error("constructor has no fixed instance output")
                        .with_identifier("RunMat:ConstructorOutputInvalid")
                        .build(),
                )
            })?;
            let slot = locals.get_mut(local.0 as usize).ok_or_else(|| {
                NativeExecutorError::from(
                    runmat_runtime::build_runtime_error(
                        "constructor instance output is out of bounds",
                    )
                    .with_identifier("RunMat:ConstructorOutputInvalid")
                    .build(),
                )
            })?;
            *slot = arena.insert(receiver.clone(), Some(&runtime))?;
        }
        for (local, value) in function
            .abi
            .fixed_inputs
            .iter()
            .zip(prepared.fixed.iter().cloned())
        {
            let slot = locals.get_mut(local.0 as usize).ok_or_else(|| {
                NativeExecutorError::Host("function ABI input local is out of bounds".into())
            })?;
            if let Some(value) = value {
                *slot = match workspace.initialize_local_owned_or_return(
                    &runtime,
                    local.0 as usize,
                    value,
                )? {
                    Some(value) => arena.insert(value, Some(&runtime))?,
                    None => NativeValueRef::NULL,
                };
            }
        }
        if let (Some(local), Some(varargin)) = (function.abi.varargin, prepared.varargin) {
            locals[local.0 as usize] = match workspace.initialize_local_owned_or_return(
                &runtime,
                local.0 as usize,
                runmat_value::Value::Cell(varargin),
            )? {
                Some(value) => arena.insert(value, Some(&runtime))?,
                None => NativeValueRef::NULL,
            };
        }
        if let Some(local) = function.abi.varargout {
            let empty = runmat_value::CellArray::new(Vec::new(), 1, 0)
                .map_err(|error| NativeExecutorError::Host(format!("varargout: {error}")))?;
            locals[local.0 as usize] = match workspace.initialize_local_owned_or_return(
                &runtime,
                local.0 as usize,
                runmat_value::Value::Cell(empty),
            )? {
                Some(value) => arena.insert(value, Some(&runtime))?,
                None => NativeValueRef::NULL,
            };
        }
        if let Some(local) = function.abi.implicit_nargin {
            locals[local.0 as usize] = arena.insert(
                runmat_value::Value::Num(prepared.nargin as f64),
                Some(&runtime),
            )?;
        }
        if let Some(local) = function.abi.implicit_nargout {
            locals[local.0 as usize] = arena.insert(
                runmat_value::Value::Num(requested_outputs as f64),
                Some(&runtime),
            )?;
        }
        for binding in &function.lexical_bindings {
            if binding.kind == runmat_native_codegen::NativeLexicalBindingKind::Transit
                && lexical_environment.is_none()
            {
                return Err(NativeExecutorError::Host(
                    "transit lexical binding has no admitted activation".into(),
                ));
            }
        }
        for (slot, local) in locals.iter_mut().enumerate() {
            if let Some(value) = workspace.initial_value_lease(&runtime, slot)? {
                *local = arena.insert_retained_lease(value, Some(&runtime))?;
                // The canonical workspace already owns this binding. Seeding
                // its native read reference is not another workspace write.
                continue;
            }
            let reference = *local;
            if !reference.is_null() {
                workspace.initialize_local_with_context(
                    &runtime,
                    slot,
                    arena.get(reference)?.clone(),
                )?;
            }
        }
        for binding in &function.lexical_bindings {
            if binding.kind == runmat_native_codegen::NativeLexicalBindingKind::Owned {
                let environment = lexical_environment.as_ref().ok_or_else(|| {
                    NativeExecutorError::Host("owned lexical binding has no activation".into())
                })?;
                if !runmat_runtime::call::lexical::reserve_owned_binding_with_context(
                    environment,
                    binding.binding,
                    &runtime,
                )? {
                    return Err(NativeExecutorError::Host(
                        "owned lexical binding was reserved twice".into(),
                    ));
                }
                let reference = *locals.get(binding.local.0 as usize).ok_or_else(|| {
                    NativeExecutorError::Host("owned lexical local is out of bounds".into())
                })?;
                if !reference.is_null() {
                    runmat_runtime::call::lexical::write_owned_binding_with_context(
                        environment,
                        binding.binding,
                        arena.get(reference)?.clone(),
                        &runtime,
                    )?;
                }
            }
        }
        if selected_inputs && supplied_inputs != 0 {
            // The selected entry moved its supplied Values into fixed locals
            // and the canonical `varargin` cell. Reuse those owners for the
            // direct ABI instead of keeping a second payload forest.
            for local in function.abi.fixed_inputs.iter().take(supplied_inputs) {
                let reference = *locals.get(local.0 as usize).ok_or_else(|| {
                    NativeExecutorError::Host("function ABI input local is out of bounds".into())
                })?;
                arena.retain(reference)?;
                argument_refs.push(reference);
            }
            if supplied_inputs > function.abi.fixed_inputs.len() {
                let local = function.abi.varargin.ok_or_else(|| {
                    NativeExecutorError::Host("native varargin local is missing".into())
                })?;
                let parent = *locals.get(local.0 as usize).ok_or_else(|| {
                    NativeExecutorError::Host("native varargin local is out of bounds".into())
                })?;
                for index in 0..supplied_inputs - function.abi.fixed_inputs.len() {
                    argument_refs.push(arena.insert_cell_element(parent, index, Some(&runtime))?);
                }
            }
        }
        let mut roots = roots::NativeRoots::default();
        roots.reserve(&runtime, locals.len())?;
        roots
            .values
            .extend(locals.iter().enumerate().map(|(slot, value)| NativeRoot {
                value: *value,
                kind: NativeRootKind::LOCAL,
                slot: slot as u32,
            }));
        let source = function.source.0;
        let mut state = Self {
            function,
            functions,
            runtime,
            inherited_catch,
            arena,
            locals,
            values: BTreeMap::new(),
            liveness: None,
            roots,
            last_error: None,
            host_failure: None,
            retirement_failure: None,
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
            contextual_index_extents: Vec::new(),
            subscript_end_receivers: Vec::new(),
            contextual_regions,
            current_request: None,
            resume_target: None,
            current_block: None,
            global_bindings,
            global_seen_revision: BTreeMap::new(),
            persistent_seen_revision: BTreeMap::new(),
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
                .enumerate()
                .map(|(index, plan)| {
                    (
                        plan.entry,
                        crate::region::OptimizedRegionPlanLease::new(
                            Arc::clone(&optimized_regions),
                            index,
                        ),
                    )
                })
                .collect(),
            region_observation: None,
            skipped_optimized_sites: BTreeSet::new(),
            vectorized_regions: 0,
            workspace: Some(workspace),
            last_expression: None,
            lexical_environment,
            _lexical_roots: lexical_roots,
        };
        for binding in state.function.inherited_storage_bindings.clone() {
            let slot = binding.local.0 as usize;
            match binding.kind {
                runmat_native_codegen::NativeInheritedStorageKind::Global => {
                    state.declare_global(slot, binding.name)?;
                }
                runmat_native_codegen::NativeInheritedStorageKind::Persistent => {
                    state.declare_persistent_for_owner(
                        slot,
                        binding.name,
                        binding.owner_function,
                    )?;
                }
            }
        }
        state.enter_block(state.function.entry)?;
        Ok((state, argument_refs))
    }
}

// Legacy input preparation owns a second, rooted Value forest. Keep those
// temporaries out of the selected invocation's state-construction frame; the
// returned root remains live until the complete HostState has been published.
#[inline(never)]
fn prepare_inputs(
    function: &runmat_native_codegen::NativeFunction,
    inputs: NativeFunctionInputs,
    captures: &[NativeCaptureInput],
) -> NativeExecutorResult<(
    Option<Vec<runmat_value::Value>>,
    runmat_runtime::call::function_abi::PreparedFunctionInputs,
    Option<ScopedValueRoots>,
)> {
    match inputs.try_into_prepared() {
        Err(arguments) => {
            let construction_values = arguments
                .iter()
                .cloned()
                .chain(captures.iter().filter_map(|capture| match capture {
                    NativeCaptureInput::Owned(capture) => Some(capture.value.clone()),
                    NativeCaptureInput::Retained { .. } => None,
                }))
                .collect();
            let roots =
                ScopedValueRoots::register(construction_values, "native_invocation_construction")?;
            let input_specs = function_input_specs(function)?;
            let prepared = prepare_function_inputs(
                &function.name,
                &arguments,
                function.abi.fixed_inputs.len(),
                function.abi.varargin.is_some(),
                &input_specs,
            )?;
            Ok((Some(arguments), prepared, Some(roots)))
        }
        Ok(prepared) => Ok((None, prepared, None)),
    }
}

pub(in crate::execute) fn function_input_specs(
    function: &runmat_native_codegen::NativeFunction,
) -> NativeExecutorResult<Vec<FunctionInputSpec<'_>>> {
    function
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
        .collect()
}
