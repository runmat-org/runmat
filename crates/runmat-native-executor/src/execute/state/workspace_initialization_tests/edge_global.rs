use super::*;

#[derive(Clone, Copy, Debug)]
enum EdgeCase {
    Normal,
    Catch,
    Await,
    Loop,
}

impl EdgeCase {
    fn argument(self, local: NativeLocalId, saved: NativeValueId) -> NativeEdgeArgument {
        match self {
            Self::Normal => NativeEdgeArgument::Value(saved),
            Self::Catch => NativeEdgeArgument::CaughtException { local },
            Self::Await => NativeEdgeArgument::AwaitResult { local },
            Self::Loop => NativeEdgeArgument::LoopIteration { local },
        }
    }

    fn transfer(
        self,
        local: NativeLocalId,
        value: NativeValueRef,
    ) -> (
        Option<(NativeLocalId, NativeValueRef)>,
        Option<NativeValueRef>,
    ) {
        match self {
            Self::Normal => (None, None),
            Self::Catch | Self::Await => (None, Some(value)),
            Self::Loop => (Some((local, value)), None),
        }
    }
}

fn edge_to_first_block(state: &mut HostState, arguments: Vec<NativeEdgeArgument>) -> NativeEdge {
    state.function.blocks[0].parameters = arguments
        .iter()
        .enumerate()
        .map(|(slot, _)| NativeBlockParameter {
            local: NativeLocalId(slot as u32),
            value: NativeValueId(42 + slot as u32),
            value_type: NativeValueType::Generic,
        })
        .collect();
    NativeEdge {
        target: NativeBlockId(0),
        arguments,
        side_effect_epoch: NativeValueId(0),
    }
}

#[derive(Clone, Copy, Debug)]
enum AliasCase {
    Global,
    Persistent,
}

fn declare_alias(state: &mut HostState, alias: AliasCase) {
    match alias {
        AliasCase::Global => state.declare_global(0, "x".into()).unwrap(),
        AliasCase::Persistent => state.declare_persistent(0, "x".into()).unwrap(),
    }
}

fn canonical_alias(state: &HostState, alias: AliasCase) -> Option<Value> {
    let authority = state.runtime.session_variable_authority();
    match alias {
        AliasCase::Global => authority.global_value("x").unwrap(),
        AliasCase::Persistent => authority
            .persistent_named_value(&state.function.name, "x")
            .unwrap(),
    }
}

#[test]
fn inherited_global_alias_survives_every_edge_without_replaying_saved_ssa() {
    for case in [
        EdgeCase::Normal,
        EdgeCase::Catch,
        EdgeCase::Await,
        EdgeCase::Loop,
    ] {
        runmat_gc::gc_test_context(|| {
            let mut state = omitted_input_state();
            let runtime = state.runtime.clone();
            let _scope = runtime.enter();
            state.declare_global(0, "x".into()).unwrap();
            let stale = state
                .arena
                .insert(Value::Num(0.0), Some(&state.runtime))
                .unwrap();
            state.set_local(0, stale).unwrap();
            state.locals.push(NativeValueRef::NULL);
            let transferred = state
                .arena
                .insert(Value::Num(19.0), Some(&state.runtime))
                .unwrap();
            state.values.insert(NativeValueId(40), stale);
            state.values.insert(NativeValueId(41), transferred);
            let edge = edge_to_first_block(
                &mut state,
                vec![
                    NativeEdgeArgument::Value(NativeValueId(40)),
                    case.argument(NativeLocalId(1), NativeValueId(41)),
                ],
            );
            runmat_runtime::workspace::session::store_global_named("x", Value::Num(1.0))
                .expect("store edge session global");
            let revision = runmat_runtime::workspace::session::global_revision();
            let (loop_iteration, transferred_value) = case.transfer(NativeLocalId(1), transferred);
            crate::execute::site::take_edge(
                &mut state,
                &edge,
                0,
                loop_iteration,
                transferred_value,
            )
            .unwrap();
            assert_eq!(
                canonical_alias(&state, AliasCase::Global),
                Some(Value::Num(1.0)),
                "{case:?}"
            );
            assert_eq!(
                runmat_runtime::workspace::session::global_revision(),
                revision,
                "{case:?}"
            );
            assert_eq!(
                state.arena.get(state.values[&NativeValueId(42)]).unwrap(),
                &Value::Num(1.0)
            );
            assert_eq!(
                state.arena.get(state.values[&NativeValueId(40)]).unwrap(),
                &Value::Num(0.0)
            );
            assert_eq!(
                state.arena.get(state.values[&NativeValueId(43)]).unwrap(),
                &Value::Num(19.0)
            );
        });
    }
}

#[test]
fn explicit_edge_overrides_publish_to_global_and_persistent_aliases() {
    for alias in [AliasCase::Global, AliasCase::Persistent] {
        for case in [EdgeCase::Catch, EdgeCase::Await, EdgeCase::Loop] {
            runmat_gc::gc_test_context(|| {
                let mut state = omitted_input_state();
                let runtime = state.runtime.clone();
                let _scope = runtime.enter();
                declare_alias(&mut state, alias);
                let incoming = state
                    .arena
                    .insert(Value::Num(19.0), Some(&state.runtime))
                    .unwrap();
                let edge = edge_to_first_block(
                    &mut state,
                    vec![case.argument(NativeLocalId(0), NativeValueId(40))],
                );
                let (loop_iteration, transferred_value) = case.transfer(NativeLocalId(0), incoming);
                crate::execute::site::take_edge(
                    &mut state,
                    &edge,
                    0,
                    loop_iteration,
                    transferred_value,
                )
                .unwrap();
                assert_eq!(
                    canonical_alias(&state, alias),
                    Some(Value::Num(19.0)),
                    "{alias:?} {case:?}"
                );
                assert_eq!(
                    state.arena.get(state.values[&NativeValueId(42)]).unwrap(),
                    &Value::Num(19.0)
                );
            });
        }
    }
}

#[test]
fn persistent_local_read_refreshes_selected_context_without_rewriting_old_ssa() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let runtime = state.runtime.clone();
        let _scope = runtime.enter();
        state.declare_persistent(0, "x".into()).unwrap();
        let old = state
            .arena
            .insert(Value::Num(0.0), Some(&state.runtime))
            .unwrap();
        state.set_local(0, old).unwrap();
        assert_eq!(
            canonical_alias(&state, AliasCase::Persistent),
            Some(Value::Num(0.0))
        );
        state.values.insert(NativeValueId(40), old);
        let authority = state.runtime.session_variable_authority();
        authority
            .store_persistent_named(&state.function.name, "x", Value::Num(1.0))
            .unwrap();
        let revision = authority.persistent_revision().unwrap();
        let other = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let _other_scope = other.enter();
        let read = crate::execute::operand::materialize_operand(
            &mut state,
            &runmat_mir::MirOperand::Local(runmat_mir::MirLocalId(0)),
        )
        .unwrap();
        assert_eq!(read, Value::Num(1.0));
        assert_eq!(
            state.arena.get(state.values[&NativeValueId(40)]).unwrap(),
            &Value::Num(0.0)
        );
        assert_eq!(authority.persistent_revision().unwrap(), revision);
        assert_eq!(
            other
                .session_variable_authority()
                .persistent_named_value(&state.function.name, "x")
                .unwrap(),
            None
        );
        let first_reference = state.local_reference(0).unwrap();
        assert_eq!(state.local_reference(0).unwrap(), first_reference);
    });
}

#[test]
fn inherited_persistent_alias_keeps_nested_write_and_old_ssa() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let runtime = state.runtime.clone();
        let _scope = runtime.enter();
        state.declare_persistent(0, "x".into()).unwrap();
        let stale = state
            .arena
            .insert(Value::Num(0.0), Some(&state.runtime))
            .unwrap();
        state.set_local(0, stale).unwrap();
        state.values.insert(NativeValueId(40), stale);
        let edge = edge_to_first_block(
            &mut state,
            vec![NativeEdgeArgument::Value(NativeValueId(40))],
        );
        let authority = state.runtime.session_variable_authority();
        authority
            .store_persistent_named(&state.function.name, "x", Value::Num(1.0))
            .unwrap();
        let revision = authority.persistent_revision().unwrap();
        crate::execute::site::take_edge(&mut state, &edge, 0, None, None).unwrap();
        assert_eq!(
            canonical_alias(&state, AliasCase::Persistent),
            Some(Value::Num(1.0))
        );
        assert_eq!(authority.persistent_revision().unwrap(), revision);
        assert_eq!(
            state.arena.get(state.values[&NativeValueId(42)]).unwrap(),
            &Value::Num(1.0)
        );
        assert_eq!(
            state.arena.get(state.values[&NativeValueId(40)]).unwrap(),
            &Value::Num(0.0)
        );
    });
}

#[test]
fn persistent_deoptimization_refreshes_only_local_not_evaluated_operand() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let runtime = state.runtime.clone();
        let _scope = runtime.enter();
        state.declare_persistent(0, "x".into()).unwrap();
        let old = state
            .arena
            .insert(Value::Num(0.0), Some(&state.runtime))
            .unwrap();
        state.set_local(0, old).unwrap();
        state.values.insert(NativeValueId(40), old);
        let terminator = state.function.blocks[0].terminator.clone();
        let mut frame = terminator.frame_state.clone();
        frame.locals.push(NativeFrameLocal {
            local: NativeLocalId(0),
            value: NativeValueId(40),
        });
        frame.operands.push(NativeValueId(40));
        state
            .runtime
            .session_variable_authority()
            .store_persistent_named(&state.function.name, "x", Value::Num(1.0))
            .unwrap();
        let materialized = state
            .materialize_deoptimization(&frame, &terminator.site)
            .unwrap();
        assert_eq!(materialized.locals[0].value, Some(Value::Num(1.0)));
        assert_eq!(materialized.operands, vec![Value::Num(0.0)]);
    });
}

#[test]
fn cleared_persistent_deoptimization_retains_tombstone_and_old_ssa_operand() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        state.declare_persistent(0, "x".into()).unwrap();
        let old = state
            .arena
            .insert(Value::Num(7.0), Some(&state.runtime))
            .unwrap();
        state.set_local(0, old).unwrap();
        state.values.insert(NativeValueId(40), old);
        let terminator = state.function.blocks[0].terminator.clone();
        let mut frame = terminator.frame_state.clone();
        frame.locals.push(NativeFrameLocal {
            local: NativeLocalId(0),
            value: NativeValueId(40),
        });
        frame.operands.push(NativeValueId(40));
        let authority = state.runtime.session_variable_authority();
        let before = state.persistent_bindings[&0].invalidation.clone();
        authority
            .clear_persistent_function(&state.function.name)
            .unwrap();
        let materialized = state
            .materialize_deoptimization(&frame, &terminator.site)
            .unwrap();
        assert!(materialized.persistent_bindings[&0].invalidated);
        assert_ne!(materialized.persistent_bindings[&0].invalidation, before);
        assert_eq!(materialized.operands, vec![Value::Num(7.0)]);
        assert_eq!(
            materialized.locals[0].value,
            Some(Value::Tensor(
                runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap()
            ))
        );
        assert_eq!(
            authority
                .persistent_named_value(&state.function.name, "x")
                .unwrap(),
            None
        );
    });
}
