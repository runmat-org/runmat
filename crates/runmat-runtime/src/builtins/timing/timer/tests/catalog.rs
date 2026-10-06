use super::*;
#[test]
fn timer_constructor_sets_defaults_and_name_values() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let value = block_on(timer_builtin(vec![
        Value::String("Name".into()),
        Value::String("solver".into()),
        Value::String("Tag".into()),
        Value::String("client".into()),
        Value::String("StartDelay".into()),
        Value::Num(0.25),
    ]))
    .expect("timer");
    let Value::HandleObject(handle) = value else {
        panic!("expected timer handle");
    };
    assert_eq!(string_property(&handle, "Name").unwrap(), "solver");
    assert_eq!(string_property(&handle, "Tag").unwrap(), "client");
    assert_eq!(numeric_property(&handle, "StartDelay").unwrap(), 0.25);
    assert_eq!(string_property(&handle, "Running").unwrap(), "off");
    assert!(crate::is_handle_valid(&handle));
}

#[test]
fn timer_family_declares_exact_integer_property_metadata() {
    for (name, capabilities) in [("timer", 2), ("timerfind", 1), ("timerfindall", 1)] {
        assert!(runmat_builtins::builtin_function_by_name(name).is_none());
        let builtin = runmat_builtins::builtin_catalog_entry_by_name(name)
            .expect("canonical timer catalog identity");
        assert_eq!(builtin.integer_capabilities.len(), capabilities, "{name}");
        assert_eq!(builtin.extensions.len(), 1, "{name}");
        assert!(builtin.integer_capabilities.iter().all(|capability| {
            capability
                .inputs
                .iter()
                .all(|input| input.classes.len() == 8)
        }));
    }
}

#[test]
fn timerfind_filters_visible_timers() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let _hidden = block_on(timer_builtin(vec![
        Value::String("Name".into()),
        Value::String("hidden".into()),
        Value::String("ObjectVisibility".into()),
        Value::String("off".into()),
    ]))
    .expect("hidden timer");
    let visible = block_on(timer_builtin(vec![
        Value::String("Name".into()),
        Value::String("visible".into()),
        Value::String("Tag".into()),
        Value::String("batch".into()),
    ]))
    .expect("visible timer");
    let found = block_on(timerfind_builtin(vec![
        Value::String("Tag".into()),
        Value::String("batch".into()),
    ]))
    .expect("timerfind");
    assert_eq!(found, visible);
    let all = block_on(timerfindall_builtin(vec![])).expect("timerfindall");
    let Value::Cell(cell) = all else {
        panic!("expected cell row of all timers");
    };
    assert_eq!((cell.rows, cell.cols), (1, 2));
}
