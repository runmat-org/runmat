use super::*;
use crate::execute::state::HostStateInput;
use runmat_native_codegen::*;
use runmat_types::{BindingId, ProgramFunctionId, ProgramPointId, ProgramSourceId, ProgramSpan};
use std::{rc::Rc, sync::Arc};

#[path = "workspace_initialization_tests/sync_completion.rs"]
mod sync_completion;

#[test]
fn declared_global_reads_follow_nested_session_writes() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let runtime = state.runtime.clone();
        let _scope = runtime.enter();
        state.declare_global(0, "x".into()).unwrap();
        let unchanged = state.local_reference(0).unwrap();
        assert_eq!(state.local_reference(0).unwrap(), unchanged);
        runmat_runtime::workspace::session::store_global_named("x", Value::Num(7.0));
        let value = crate::execute::operand::materialize_operand(
            &mut state,
            &runmat_mir::MirOperand::Local(runmat_mir::MirLocalId(0)),
        )
        .unwrap();
        assert_eq!(value, Value::Num(7.0));
        assert_eq!(state.assignment_root(0, false).unwrap(), Value::Num(7.0));

        let reference = state.arena.insert(Value::Num(9.0));
        state.set_local(0, reference).unwrap();
        assert_eq!(
            runmat_runtime::workspace::session::global_value("x"),
            Some(Value::Num(9.0))
        );
        state.set_local(0, NativeValueRef::NULL).unwrap();
        let restored = state.local_reference(0).unwrap();
        assert_eq!(state.arena.get(restored).unwrap(), &Value::Num(9.0));
    });
}

#[test]
fn missing_canonical_global_replaces_an_old_frame_snapshot_with_empty() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let runtime = state.runtime.clone();
        let _scope = runtime.enter();
        state.declare_global(0, "x".into()).unwrap();
        let old = state.arena.insert(Value::Num(9.0));
        state.set_local(0, old).unwrap();
        let refreshed = state
            .replace_local_from_global(
                0,
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
        let runtime = state.runtime.clone();
        let _scope = runtime.enter();
        state.declare_global(0, "x".into()).unwrap();
        let old = state.arena.insert(Value::Num(1.0));
        state.set_local(0, old).unwrap();
        state.values.insert(NativeValueId(1), old);
        let terminator = state.function.blocks[0].terminator.clone();
        let mut frame = terminator.frame_state.clone();
        frame.locals.push(NativeFrameLocal {
            local: NativeLocalId(0),
            value: NativeValueId(1),
        });
        frame.operands.push(NativeValueId(1));

        runmat_runtime::workspace::session::store_global_named("x", Value::Num(7.0));
        let materialized = state
            .materialize_deoptimization(&frame, &terminator.site)
            .unwrap();
        assert_eq!(materialized.locals[0].value, Some(Value::Num(7.0)));
        assert_eq!(materialized.operands, vec![Value::Num(1.0)]);
    });
}

fn omitted_input_state() -> HostState {
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
        id: ProgramFunctionId(0),
        source: ProgramSourceId(0),
        name: "omitted_input".into(),
        source_role: runmat_types::FunctionSourceRole::Unknown,
        class_method_owner: None,
        capabilities: Default::default(),
        abi: NativeFunctionAbi {
            fixed_inputs: vec![NativeLocalId(0)],
            varargin: None,
            fixed_outputs: vec![],
            varargout: None,
            implicit_nargin: None,
            implicit_nargout: None,
        },
        argument_validations: vec![],
        locals: vec![NativeLocalMetadata {
            id: NativeLocalId(0),
            binding: Some(BindingId(0)),
            name: Some("x".into()),
            kind: NativeLocalKind::Binding,
        }],
        entry: NativeBlockId(0),
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
        arguments: vec![],
        requested_outputs: 0,
        runtime: RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        )),
        program_capture: None,
        captures: vec![],
        deoptimization: Default::default(),
        interpreter_resume_points: Default::default(),
        coverage_sites: Default::default(),
        osr_point: None,
        optimized_regions: Default::default(),
        workspace: None,
    })
    .unwrap()
    .0
}

fn missing_error(state: &mut HostState) {
    assert!(
        matches!(state.assignment_root(0, true), Err(NativeExecutorError::Runtime(error)) if error.identifier() == Some("RunMat:NotEnoughInputs"))
    );
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
            .insert(Value::Struct(runmat_value::StructValue::new()));
        state.set_local(0, reference).unwrap();
        assert!(matches!(
            state.assignment_root(0, true).unwrap(),
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
            .synchronize_local(0, Value::Struct(runmat_value::StructValue::new()))
            .unwrap();
        state.reconcile_workspace().unwrap();
        assert!(matches!(
            state.assignment_root(0, true).unwrap(),
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
                future: Box::pin(async move {
                    service.assign("x", Value::Num(42.0))?;
                    if fail {
                        Err(runmat_runtime::runtime_error::semantic_error(
                            "WorkspaceCompletionFixture",
                            "failure after workspace assignment",
                        )
                        .into())
                    } else {
                        Ok(crate::execute::call_suspension::CallCompletion::Values(
                            vec![],
                        ))
                    }
                }),
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
            assert_eq!(state.assignment_root(0, false).unwrap(), Value::Num(42.0));
            assert!(state.missing_input_locals.is_empty());
            assert!(!state.workspace.as_ref().unwrap().needs_reconciliation());
        }
    });
}
