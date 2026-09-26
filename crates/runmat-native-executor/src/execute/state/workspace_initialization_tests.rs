use super::*;
use crate::execute::state::HostStateInput;
use runmat_native_codegen::*;
use runmat_types::{BindingId, ProgramFunctionId, ProgramPointId, ProgramSourceId, ProgramSpan};
use std::{rc::Rc, sync::Arc};

#[path = "workspace_initialization_tests/sync_completion.rs"]
mod sync_completion;

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

fn missing_error(state: &HostState) {
    assert!(
        matches!(state.assignment_root(0, true), Err(NativeExecutorError::Runtime(error)) if error.identifier() == Some("RunMat:NotEnoughInputs"))
    );
}

#[test]
fn non_null_write_initializes_an_omitted_input_and_deopt_metadata() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        missing_error(&state);
        state.set_local(0, NativeValueRef::NULL).unwrap();
        missing_error(&state);
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
        missing_error(&state);
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
