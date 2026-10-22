use super::*;

#[test]
fn hgload_applies_property_overrides_and_returns_old_values() {
    let _guard = setup();
    let fig = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("Before".into()),
        ]),
    )
    .expect("figure");
    let path = temp_path("hgload");
    let _ = std::fs::remove_file(&path);
    block_on(hgsave_builtin(vec![
        Value::Num(fig),
        Value::String(path.to_string_lossy().into_owned().into()),
    ]))
    .expect("hgsave");
    clear_figure(None).expect("clear");
    let mut overrides = runmat_value::UnboundedStructValueBuilder::new();
    overrides.insert("Name", Value::String("After".into()));
    overrides.insert("NotAProperty", Value::Num(1.0));
    let result = block_on(execute_hgload(
        None,
        Some(2),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String(path.to_string_lossy().into_owned().into()),
            Value::Struct(overrides.finish()),
        ]),
    ))
    .expect("hgload");
    let outputs = result.into_output_storage();
    assert_eq!(outputs.len(), 2);
    let Value::Cell(cell) = &outputs[1] else {
        panic!("expected old values cell");
    };
    let Value::Struct(old) = &cell.data()[0] else {
        panic!("expected old values struct");
    };
    assert_eq!(
        old.fields.get("Name"),
        Some(&Value::String("Before".into()))
    );
    assert!(!old.fields.contains_key("NotAProperty"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn hgload_output_admission_failure_rolls_back_imported_figures() {
    let _guard = setup();
    let figure = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("source figure");
    let path = temp_path("hgload_output_rollback");
    let _ = std::fs::remove_file(&path);
    block_on(hgsave_builtin(vec![
        Value::Num(figure),
        Value::String(path.to_string_lossy().into_owned().into()),
    ]))
    .expect("save archive");
    reset_plot_state();
    let before = figure_handles();
    block_on(execute_hgload(
        None,
        Some(3),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            path.to_string_lossy().into_owned().into(),
        )]),
    ))
    .expect_err("three outputs reject");
    assert_eq!(figure_handles(), before);
    let _ = std::fs::remove_file(path);
}
