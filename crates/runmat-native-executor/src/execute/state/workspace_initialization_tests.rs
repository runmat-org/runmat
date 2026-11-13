use super::*;
use crate::execute::state::HostStateInput;
use runmat_native_codegen::*;
use runmat_types::AssignmentCreationPolicy;
use runmat_types::{BindingId, ProgramFunctionId, ProgramPointId, ProgramSourceId, ProgramSpan};
use std::{rc::Rc, sync::Arc};

#[path = "workspace_initialization_tests/edge_global.rs"]
mod edge_global;
#[path = "workspace_initialization_tests/numeric_region_inputs.rs"]
mod numeric_region_inputs;
#[path = "workspace_initialization_tests/operation_completion.rs"]
mod operation_completion;

#[test]
fn first_global_declaration_replaces_a_stale_local_with_canonical_empty() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let runtime = state.runtime.clone();
        let _scope = runtime.enter();
        let stale = state
            .arena
            .insert(Value::Num(42.0), Some(&state.runtime))
            .unwrap();
        state.set_local(0, stale).unwrap();

        state.declare_global(0, "fresh_global".into()).unwrap();
        let declared = state.local_reference(0).unwrap();
        assert_eq!(state.arena.get(declared).unwrap(), &empty_workspace_value());
    });
}

#[test]
fn lexical_reads_reuse_unchanged_generations_and_preserve_prior_operands() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let (environment, _roots) =
            runmat_runtime::call::lexical::allocate_environment(None).unwrap();
        runmat_runtime::call::lexical::reserve_owned_binding(&environment, BindingId(0)).unwrap();
        state.function.lexical_bindings.push(NativeLexicalBinding {
            binding: BindingId(0),
            local: NativeLocalId(0),
            kind: NativeLexicalBindingKind::Owned,
        });
        state.lexical_environment = Some(environment.clone());
        runmat_runtime::call::lexical::write_owned_binding(
            &environment,
            BindingId(0),
            Value::Num(7.0),
        )
        .unwrap();
        let original = state.local_reference(0).unwrap();
        assert_eq!(state.local_reference(0).unwrap(), original);
        runmat_runtime::call::lexical::write_owned_binding(
            &environment,
            BindingId(0),
            Value::Num(9.0),
        )
        .unwrap();
        let replacement = state.local_reference(0).unwrap();
        assert_ne!(replacement, original);
        assert_eq!(state.local_reference(0).unwrap(), replacement);
        assert_eq!(state.arena.get(original).unwrap(), &Value::Num(7.0));
        assert_eq!(state.arena.get(replacement).unwrap(), &Value::Num(9.0));
        runmat_runtime::call::lexical::remove_owned_binding(&environment, BindingId(0)).unwrap();
        assert!(state.local_reference(0).unwrap().is_null());
        assert_eq!(state.arena.get(original).unwrap(), &Value::Num(7.0));
    });
}

#[test]
fn declared_global_reads_follow_nested_session_writes() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let runtime = state.runtime.clone();
        let _scope = runtime.enter();
        state.declare_global(0, "x".into()).unwrap();
        let unchanged = state.local_reference(0).unwrap();
        assert_eq!(state.local_reference(0).unwrap(), unchanged);
        runmat_runtime::workspace::session::store_global_named("x", Value::Num(7.0))
            .expect("store nested session global");
        let value = crate::execute::operand::materialize_operand(
            &mut state,
            &runmat_mir::MirOperand::Local(runmat_mir::MirLocalId(0)),
        )
        .unwrap();
        assert_eq!(value, Value::Num(7.0));
        assert_eq!(
            state
                .assignment_root(0, AssignmentCreationPolicy::ExistingOnly)
                .unwrap(),
            Value::Num(7.0)
        );

        let reference = state
            .arena
            .insert(Value::Num(9.0), Some(&state.runtime))
            .unwrap();
        state.set_local(0, reference).unwrap();
        assert_eq!(
            runmat_runtime::workspace::session::global_value("x"),
            Some(Value::Num(9.0))
        );
        let authority = runtime.session_variable_authority();
        authority
            .store_global_with_context(&runtime, "h", Value::Num(17.0))
            .unwrap();
        let target = runmat_runtime::workspace::GlobalBindingTarget::Named(
            runmat_runtime::workspace::WorkspaceName::unbounded("h"),
        );
        let prepared = state
            .global_bindings
            .prepare_target(&runtime, 0, &target)
            .unwrap();
        prepared.checkpoint().unwrap();
        prepared.commit();
        let retargeted = state.local_reference(0).unwrap();
        assert_eq!(state.arena.get(retargeted).unwrap(), &Value::Num(17.0));
        let updated = state
            .arena
            .insert(Value::Num(19.0), Some(&state.runtime))
            .unwrap();
        state.set_local(0, updated).unwrap();
        assert_eq!(authority.global_value("h").unwrap(), Some(Value::Num(19.0)));
        assert_eq!(authority.global_value("x").unwrap(), Some(Value::Num(9.0)));
        state.set_local(0, NativeValueRef::NULL).unwrap();
        let restored = state.local_reference(0).unwrap();
        assert_eq!(state.arena.get(restored).unwrap(), &Value::Num(19.0));
    });
}

#[test]
fn missing_canonical_global_replaces_an_old_frame_snapshot_with_empty() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let runtime = state.runtime.clone();
        let _scope = runtime.enter();
        state.declare_global(0, "x".into()).unwrap();
        let old = state
            .arena
            .insert(Value::Num(9.0), Some(&state.runtime))
            .unwrap();
        state.set_local(0, old).unwrap();
        let refreshed = state
            .replace_local_from_global(
                0,
                runmat_runtime::workspace::GlobalBindingTarget::Named(
                    runmat_runtime::workspace::WorkspaceName::unbounded("x"),
                ),
                None,
                runmat_runtime::workspace::session::GlobalRevision::default(),
            )
            .unwrap();
        assert_eq!(
            state.arena.get(refreshed).unwrap(),
            &empty_workspace_value()
        );
        assert_ne!(refreshed, old);
    });
}

#[test]
fn global_deoptimization_refreshes_only_the_alias_not_a_shared_ssa_operand() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        state.function.id = runmat_types::FunctionId(77);
        let runtime = state.runtime.clone();
        let _scope = runtime.enter();
        state.declare_global(0, "x".into()).unwrap();
        let old = state
            .arena
            .insert(Value::Num(1.0), Some(&state.runtime))
            .unwrap();
        state.set_local(0, old).unwrap();
        state.values.insert(NativeValueId(1), old);
        let terminator = state.function.blocks[0].terminator.clone();
        state.function.expected_sites.push(terminator.site.clone());
        let resume_request = state
            .resume_request_for_block(state.function.entry)
            .unwrap();
        assert_eq!(
            resume_request.function, 0,
            "resume sites retain portable owner zero, not local coordinate 77"
        );
        let resume = runmat_runtime::native::NativeResumeState {
            function: resume_request.function,
            block: resume_request.block,
            position: resume_request.position,
            phase: resume_request.phase.0,
            ordinal: resume_request.ordinal,
            ..Default::default()
        };
        state.prepare_resume(resume).unwrap();
        assert!(state
            .prepare_resume(runmat_runtime::native::NativeResumeState {
                function: 77,
                ..resume
            })
            .is_err());
        let mut frame = terminator.frame_state.clone();
        frame.locals.push(NativeFrameLocal {
            local: NativeLocalId(0),
            value: NativeValueId(1),
        });
        frame.operands.push(NativeValueId(1));

        runmat_runtime::workspace::session::store_global_named("x", Value::Num(7.0))
            .expect("store deoptimization session global");
        let materialized = state
            .materialize_deoptimization(&frame, &terminator.site)
            .unwrap();
        assert_eq!(materialized.function, runmat_types::FunctionId(77));
        assert_eq!(materialized.site.point.function, ProgramFunctionId(0));
        assert_eq!(materialized.locals[0].value, Some(Value::Num(7.0)));
        assert_eq!(materialized.operands, vec![Value::Num(1.0)]);
        frame.point.function = ProgramFunctionId(77);
        assert!(state
            .materialize_deoptimization(&frame, &terminator.site)
            .is_err());
    });
}

#[test]
fn inherited_persistent_local_delegates_to_ancestor_named_storage() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let runtime = state.runtime.clone();
        let _scope = runtime.enter();
        let authority = runtime.session_variable_authority();
        authority
            .store_persistent_named("outer", "P", Value::Num(5.0))
            .unwrap();
        authority
            .store_persistent_slot("outer", 0, Value::Num(33.0))
            .unwrap();

        state
            .declare_persistent_for_owner(0, "P".into(), "outer".into())
            .unwrap();
        let reference = state.local_reference(0).unwrap();
        assert_eq!(state.arena.get(reference).unwrap(), &Value::Num(5.0));

        let updated = state
            .arena
            .insert(Value::Num(7.0), Some(&state.runtime))
            .unwrap();
        state.set_local(0, updated).unwrap();
        assert_eq!(
            authority.persistent_named_value("outer", "P").unwrap(),
            Some(Value::Num(7.0))
        );
        assert_eq!(
            authority.persistent_slot_value("outer", 0).unwrap(),
            Some(Value::Num(33.0))
        );

        authority
            .store_persistent_named("outer", "P", Value::Num(8.0))
            .unwrap();
        let refreshed = state.local_reference(0).unwrap();
        assert_eq!(state.arena.get(refreshed).unwrap(), &Value::Num(8.0));
    });
}

#[test]
fn returned_child_capture_cannot_replay_older_canonical_storage() {
    runmat_gc::gc_test_context(|| {
        let mut global_state = omitted_input_state();
        let runtime = global_state.runtime.clone();
        let _scope = runtime.enter();
        let authority = runtime.session_variable_authority();
        global_state.declare_global(0, "G".into()).unwrap();
        authority
            .store_global_with_context(&runtime, "G", Value::Num(9.0))
            .unwrap();
        global_state
            .apply_lexical_captures(vec![runmat_runtime::call::lexical::LexicalCapture {
                binding: BindingId(0),
                value: Value::Num(1.0),
            }])
            .unwrap();
        assert_eq!(authority.global_value("G").unwrap(), Some(Value::Num(9.0)));

        let mut persistent_state = omitted_input_state_with(runtime.clone(), None);
        persistent_state
            .declare_persistent_for_owner(0, "P".into(), "outer".into())
            .unwrap();
        authority
            .store_persistent_named_with_context(&runtime, "outer", "P", Value::Num(9.0))
            .unwrap();
        persistent_state
            .apply_lexical_captures(vec![runmat_runtime::call::lexical::LexicalCapture {
                binding: BindingId(0),
                value: Value::Num(1.0),
            }])
            .unwrap();
        assert_eq!(
            authority.persistent_named_value("outer", "P").unwrap(),
            Some(Value::Num(9.0))
        );
        authority.clear_persistent_function("outer").unwrap();
        persistent_state
            .apply_lexical_captures(vec![runmat_runtime::call::lexical::LexicalCapture {
                binding: BindingId(0),
                value: Value::Num(1.0),
            }])
            .unwrap();
        assert_eq!(
            authority.persistent_named_value("outer", "P").unwrap(),
            None,
            "a returned child capture cannot resurrect a cleared persistent"
        );
    });
}

fn omitted_input_state() -> HostState {
    omitted_input_state_with(
        RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        )),
        None,
    )
}

fn omitted_input_state_with(
    runtime: RuntimeContext,
    workspace: Option<crate::execute::workspace::NativeWorkspaceInput>,
) -> HostState {
    local_state_with(runtime, workspace, vec![NativeLocalId(0)])
}

fn local_state_with(
    runtime: RuntimeContext,
    workspace: Option<crate::execute::workspace::NativeWorkspaceInput>,
    fixed_inputs: Vec<NativeLocalId>,
) -> HostState {
    let point = ProgramPointId {
        function: ProgramFunctionId(0),
        block: 0,
        position: 0,
    };
    let source = NativeSourceLocation {
        source: ProgramSourceId(0),
        span: ProgramSpan { start: 0, end: 0 },
    };
    let function = NativeFunction {
        spawn_safety: runmat_types::SpawnSafetyFact::SpawnSafe,
        lexical_bindings: Vec::new(),
        inherited_storage_bindings: Vec::new(),
        creates_lexical_environment: false,
        id: runmat_types::FunctionId(0),
        portable_id: ProgramFunctionId(0),
        mir_body_digest: runmat_execution::Digest::sha256(b"fixture:native-omitted-input"),
        source: ProgramSourceId(0),
        name: "omitted_input".into(),
        source_role: runmat_types::FunctionSourceRole::Unknown,
        lexical_parent: None,
        class_method_owner: None,
        capabilities: Default::default(),
        abi: NativeFunctionAbi {
            fixed_inputs,
            varargin: None,
            fixed_outputs: vec![],
            varargout: None,
            implicit_nargin: None,
            implicit_nargout: None,
        },
        argument_validations: vec![],
        binding_locals: vec![runmat_native_codegen::NativeBindingLocal {
            binding: BindingId(0),
            local: NativeLocalId(0),
        }],
        closure_captures: vec![],
        locals: vec![NativeLocalMetadata {
            id: NativeLocalId(0),
            binding: Some(BindingId(0)),
            name: Some("x".into()),
            kind: NativeLocalKind::Binding,
        }],
        entry: NativeBlockId(0),
        expression_continuation_geometry:
            runmat_native_codegen::NativeExpressionContinuationGeometry {
                region_capacity: 16,
                retained_input_capacity: 16,
            },
        operation_contracts: vec![],
        expected_sites: vec![],
        blocks: vec![NativeBlock {
            id: NativeBlockId(0),
            catch_scopes: vec![],
            parameters: vec![],
            side_effect_epoch: NativeValueId(0),
            region_boundaries: vec![],
            instructions: vec![],
            terminator: NativeTerminator {
                site: NativeMirSite {
                    point,
                    phase: NativeSitePhase::Terminator,
                    ordinal: 0,
                    construct: runmat_mir::MirConstructKind::Return,
                },
                source: source.clone(),
                class: runmat_mir::NativeLoweringClass::NativeOperation,
                effects: Default::default(),
                capabilities: Default::default(),
                safepoint: None,
                kind: NativeTerminatorKind::Return { values: vec![] },
                frame_state: NativeFrameState {
                    point,
                    source,
                    locals: vec![],
                    operands: vec![],
                    side_effect_epoch: NativeValueId(0),
                },
            },
        }],
    };
    HostState::new(HostStateInput {
        functions: Arc::new(vec![function.clone()]),
        function,
        inputs: NativeFunctionInputs::legacy(vec![]),
        requested_outputs: 0,
        runtime,
        program_capture: None,
        captures: vec![],
        lexical_environment: None,
        deoptimization: Default::default(),
        interpreter_resume_points: Default::default(),
        coverage_sites: Default::default(),
        osr_point: None,
        optimized_regions: Default::default(),
        workspace,
    })
    .unwrap()
    .0
}

#[test]
fn native_function_frame_owns_exact_arity_even_when_fixed_input_is_omitted() {
    runmat_gc::gc_test_context(|| {
        let state = omitted_input_state();
        assert_eq!(
            state.runtime.invocation_arity(),
            Some(runmat_runtime::context::RuntimeInvocationArity::function(
                0, 0
            ))
        );
        assert_eq!(state.supplied_inputs, 0);
    });
}

fn missing_error(state: &mut HostState) {
    assert!(
        matches!(state.assignment_root(0, AssignmentCreationPolicy::CreateStructFieldPath), Err(NativeExecutorError::Runtime(error)) if error.identifier() == Some("RunMat:NotEnoughInputs"))
    );
}

#[test]
fn indexed_root_creation_initializes_only_absent_non_input_numeric_bindings() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let mut state = local_state_with(runtime, None, Vec::new());
        assert!(state
            .assignment_root(0, AssignmentCreationPolicy::ExistingOnly)
            .is_err());
        let Value::Tensor(empty) = state
            .assignment_root(0, AssignmentCreationPolicy::CreateArrayByIndex)
            .expect("absent indexed root starts as an empty numeric array")
        else {
            panic!("indexed creation must use the numeric array carrier")
        };
        assert_eq!(empty.shape(), &[0, 0]);
        assert_eq!(empty.numeric_dtype(), runmat_value::NumericDType::F64);
        assert!(
            state.locals[0].is_null(),
            "planning does not publish a binding"
        );

        let existing = state
            .arena
            .insert(Value::Num(7.0), Some(&state.runtime))
            .unwrap();
        state.set_local(0, existing).unwrap();
        assert_eq!(
            state
                .assignment_root(0, AssignmentCreationPolicy::CreateArrayByIndex)
                .unwrap(),
            Value::Num(7.0),
            "creation policy cannot replace an existing binding"
        );

        let mut omitted = omitted_input_state();
        assert!(matches!(
            omitted.assignment_root(0, AssignmentCreationPolicy::CreateArrayByIndex),
            Err(NativeExecutorError::Runtime(error))
                if error.identifier() == Some("RunMat:NotEnoughInputs")
        ));
    });
}

#[test]
fn non_null_write_initializes_an_omitted_input_and_deopt_metadata() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        missing_error(&mut state);
        state.set_local(0, NativeValueRef::NULL).unwrap();
        missing_error(&mut state);
        let reference = state
            .arena
            .insert(
                Value::Struct(runmat_value::StructValue::new()),
                Some(&state.runtime),
            )
            .unwrap();
        state.set_local(0, reference).unwrap();
        assert!(matches!(
            state
                .assignment_root(0, AssignmentCreationPolicy::CreateStructFieldPath)
                .unwrap(),
            Value::Struct(_)
        ));
        let terminator = state.function.blocks[0].terminator.clone();
        let frame = state
            .materialize_deoptimization(&terminator.frame_state, &terminator.site)
            .unwrap();
        assert!(frame.missing_input_locals.is_empty());
        assert_eq!(
            frame.supplied_inputs, 0,
            "initialization does not fabricate nargin"
        );
    });
}

#[test]
fn workspace_reconciliation_records_initialization_but_not_absence() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        state.reconcile_workspace().unwrap();
        missing_error(&mut state);
        state
            .workspace
            .as_ref()
            .unwrap()
            .synchronize_local_with_context(
                &state.runtime,
                0,
                Value::Struct(runmat_value::StructValue::new()),
            )
            .unwrap();
        state.reconcile_workspace().unwrap();
        assert!(matches!(
            state
                .assignment_root(0, AssignmentCreationPolicy::CreateStructFieldPath)
                .unwrap(),
            Value::Struct(_)
        ));
        assert!(state.missing_input_locals.is_empty());
    });
}

#[test]
fn suspended_call_completion_reconciles_workspace_on_success_and_error() {
    runmat_gc::gc_test_context(|| {
        for fail in [false, true] {
            let mut state = omitted_input_state();
            let service = state.workspace.as_ref().unwrap().service();
            let call_runtime = state.runtime.clone();
            let request = runmat_runtime::native::NativeSiteRequest {
                function: 0,
                block: 0,
                position: 0,
                phase: runmat_runtime::native::NativeSitePhase::RVALUE,
                ordinal: 0,
                reserved: 0,
            };
            state.pending_call = Some(crate::execute::call_suspension::PendingCall {
                continuation: 1,
                generation: 1,
                request,
                embedded: None,
                retained_roots: Vec::new(),
                future: crate::execute::call_suspension::admit_completion_with(
                    &state,
                    move || async move {
                        service.assign_with_context(&call_runtime, "x", Value::Num(42.0))?;
                        if fail {
                            Err(runmat_runtime::runtime_error::semantic_error(
                                "WorkspaceCompletionFixture",
                                "failure after workspace assignment",
                            )
                            .into())
                        } else {
                            Ok(
                                crate::execute::call_suspension::CallCompletion::RootedArguments(
                                    runmat_runtime::execution::RootedValues::new(vec![])
                                        .map_err(|error| error.into_runtime_error())?,
                                ),
                            )
                        }
                    },
                )
                .unwrap(),
            });
            let result = futures::executor::block_on(crate::execute::call_suspension::complete(
                &mut state, 1, 1,
            ));
            if fail {
                assert!(matches!(result, Err(NativeExecutorError::Runtime(error))
                    if error.identifier() == Some("RunMat:WorkspaceCompletionFixture")));
            } else {
                assert_eq!(result.unwrap(), request);
            }
            assert_eq!(
                state
                    .assignment_root(0, AssignmentCreationPolicy::ExistingOnly)
                    .unwrap(),
                Value::Num(42.0)
            );
            assert!(state.missing_input_locals.is_empty());
            assert!(!state.workspace.as_ref().unwrap().needs_reconciliation());
        }
    });
}

#[test]
fn suspended_contextual_call_reconciles_a_new_canonical_revision() {
    runmat_gc::gc_test_context(|| {
        for fail in [false, true] {
            let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
            let mut initial = runmat_runtime::workspace::WorkspaceDelta::new();
            initial.assign("x", Value::Num(1.0)).unwrap();
            owner.apply(initial).unwrap();
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ));
            let foreground = Rc::new(
                runmat_runtime::workspace::BaseWorkspaceProjectionService::new_for_context(
                    owner.begin_projection().unwrap(),
                    &runtime,
                ),
            );
            let canonical: Rc<dyn runmat_runtime::context::RuntimeWorkspaceService> = foreground;
            let ports = runtime.service_ports().clone().with_workspace(canonical);
            let runtime = runtime.with_service_ports(ports);
            let workspace = crate::execute::workspace::NativeWorkspaceInput::from_test_values(
                std::collections::BTreeMap::from([("x".into(), Value::Num(1.0))]),
                std::collections::BTreeMap::from([(BindingId(0), "x".into())]),
                vec![crate::execute::workspace::NativeWorkspaceBinding {
                    binding: BindingId(0),
                    name: "x".into(),
                }],
                Vec::new(),
            );
            let mut state = omitted_input_state_with(runtime, Some(workspace));

            let request = runmat_runtime::native::NativeSiteRequest {
                function: 0,
                block: 0,
                position: 0,
                phase: runmat_runtime::native::NativeSitePhase::RVALUE,
                ordinal: 0,
                reserved: 0,
            };
            let call_runtime = state.runtime.clone();
            let reached = Rc::new(std::cell::Cell::new(false));
            let reached_call = Rc::clone(&reached);
            let (release, gate) = futures::channel::oneshot::channel::<()>();
            state.pending_call = Some(crate::execute::call_suspension::PendingCall {
                continuation: 1,
                generation: 1,
                request,
                embedded: None,
                retained_roots: Vec::new(),
                future: crate::execute::call_suspension::admit_completion_with(
                    &state,
                    move || async move {
                        reached_call.set(true);
                        gate.await.map_err(|_| {
                            runmat_runtime::runtime_error::semantic_error(
                                "WorkspaceRevisionFixture",
                                "checkpoint gate was abandoned",
                            )
                        })?;
                        runmat_runtime::call_builtin_async_in_context(
                            &call_runtime,
                            "plus",
                            &[Value::Num(1.0), Value::Num(2.0)],
                        )
                        .await?;
                        if fail {
                            Err(runmat_runtime::runtime_error::semantic_error(
                                "WorkspaceRevisionFixture",
                                "failure after canonical reconciliation",
                            )
                            .into())
                        } else {
                            Ok(
                                crate::execute::call_suspension::CallCompletion::RootedArguments(
                                    runmat_runtime::execution::RootedValues::new(vec![])
                                        .map_err(|error| error.into_runtime_error())?,
                                ),
                            )
                        }
                    },
                )
                .unwrap(),
            });
            let mut completion =
                Box::pin(crate::execute::call_suspension::complete(&mut state, 1, 1));
            let mut context = std::task::Context::from_waker(futures::task::noop_waker_ref());
            assert!(matches!(
                std::future::Future::poll(completion.as_mut(), &mut context),
                std::task::Poll::Pending
            ));
            assert!(reached.get(), "native call reached its suspension gate");

            let mut callback = owner.begin_projection().unwrap();
            callback.assign("x", Value::Num(42.0)).unwrap();
            callback.publish().unwrap();
            release.send(()).expect("resume suspended native call");
            let result = futures::executor::block_on(completion.as_mut());
            drop(completion);
            if fail {
                assert!(matches!(result, Err(NativeExecutorError::Runtime(error))
                    if error.identifier() == Some("RunMat:WorkspaceRevisionFixture")));
            } else {
                assert_eq!(result.unwrap(), request);
            }
            assert_eq!(
                state
                    .assignment_root(0, AssignmentCreationPolicy::ExistingOnly)
                    .unwrap(),
                Value::Num(42.0)
            );
            assert!(!state.workspace.as_ref().unwrap().needs_reconciliation());
        }
    });
}
