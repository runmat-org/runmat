use super::*;

#[test]
fn readtimetable_selects_first_time_typed_variable_not_first_column() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let path = unique_path("readtimetable_later_time_column").with_extension("csv");
    fs::write(
        &path,
        "Count,When,Later\n10,2026-07-01,2026-08-01\n20,2026-07-02,2026-08-02\n",
    )
    .unwrap();
    let types = Value::StringArray(
        StringArray::new(
            vec!["uint64".into(), "datetime".into(), "datetime".into()],
            vec![1, 3],
        )
        .unwrap(),
    );

    let result = block_on(readtimetable_builtin(
        Value::from(path.to_string_lossy().to_string()),
        vec![Value::from("VariableTypes"), types],
    ));
    let _ = fs::remove_file(&path);
    let timetable = object(result.expect("later datetime column must supply row times"));
    assert_eq!(
        table_variable_names_from_object(&timetable).unwrap(),
        vec!["Count".to_string(), "Later".to_string()]
    );
    assert!(matches!(
        timetable_row_times(&timetable).unwrap(),
        Some(Value::Object(times)) if times.is_class(runmat_types::standard::DATETIME)
    ));
    match table_member_get(&timetable, &Value::from("Count")).unwrap() {
        Value::Tensor(values) => {
            assert_eq!(
                values.integer_storage(),
                Some(&IntegerStorage::U64(vec![10, 20]))
            );
        }
        other => panic!("expected exact integer variable, got {other:?}"),
    }
}
