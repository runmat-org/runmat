use super::*;

fn two_input_plan() -> crate::region::OptimizedRegionPlan {
    let mut plan = crate::region::placement_test_support::plan();
    plan.inputs.push(runmat_types::RegionValueId {
        function: plan.region.function,
        local: 1,
    });
    plan
}

#[test]
fn omitted_input_makes_numeric_region_ineligible_without_consulting_placement() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let plan = crate::region::placement_test_support::plan();
        assert!(state.optimized_region_inputs(&plan).unwrap().is_none());
        state.optimized_regions.insert(plan.entry, plan);
        let mut exit = runmat_runtime::native::NativeExit::completed(0);
        let outcome = crate::execute::region::checkpoint(
            &mut state,
            runmat_runtime::native::NativeSiteRequest {
                function: 0,
                block: 0,
                position: 0,
                phase: runmat_runtime::native::NativeSitePhase::RVALUE,
                ordinal: 0,
                reserved: 0,
            },
            &mut exit,
        )
        .unwrap();
        assert!(outcome.is_none());
        assert!(state.region_observation.is_none());
        assert_eq!(state.vectorized_regions(), 0);
    });
}

#[test]
fn missing_input_does_not_mask_invalid_slot_or_stale_nonnull_reference() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let plan = two_input_plan();
        assert!(matches!(
            state.optimized_region_inputs(&plan),
            Err(NativeExecutorError::Host(message)) if message == "native local is out of bounds"
        ));

        let stale = state.arena.insert(Value::Num(3.0));
        state.arena.release(stale).unwrap();
        state.locals.push(stale);
        assert!(matches!(
            state.optimized_region_inputs(&plan),
            Err(NativeExecutorError::StaleValue)
        ));
        assert_eq!(state.vectorized_regions(), 0);
    });
}

#[test]
fn numeric_region_reads_current_global_instead_of_old_local_snapshot() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        state.declare_global(0, "x".into()).unwrap();
        let old = state.arena.insert(Value::Num(1.0));
        state.set_local(0, old).unwrap();
        state.values.insert(NativeValueId(40), old);
        state
            .runtime
            .session_variable_authority()
            .store_global("x", Value::Num(7.0))
            .unwrap();

        let plan = crate::region::placement_test_support::plan();
        assert_eq!(
            state.optimized_region_inputs(&plan).unwrap(),
            Some(vec![&Value::Num(7.0)])
        );
        assert_eq!(
            state.arena.get(state.values[&NativeValueId(40)]).unwrap(),
            &Value::Num(1.0)
        );
    });
}

#[test]
fn numeric_region_reads_persistent_clear_and_reinitialization_from_authority() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        state.declare_persistent(0, "x".into()).unwrap();
        let authority = state.runtime.session_variable_authority();
        let plan = crate::region::placement_test_support::plan();
        authority
            .store_persistent_named(&state.function.name, "x", Value::Num(4.0))
            .unwrap();
        assert_eq!(
            state.optimized_region_inputs(&plan).unwrap(),
            Some(vec![&Value::Num(4.0)])
        );
        authority
            .clear_persistent_function(&state.function.name)
            .unwrap();
        assert_eq!(
            state.optimized_region_inputs(&plan).unwrap(),
            Some(vec![&empty_workspace_value()])
        );
        assert!(state.persistent_bindings[&0].invalidated);
        authority
            .store_persistent_named(&state.function.name, "x", Value::Num(9.0))
            .unwrap();
        assert_eq!(
            state.optimized_region_inputs(&plan).unwrap(),
            Some(vec![&Value::Num(9.0)])
        );
    });
}
