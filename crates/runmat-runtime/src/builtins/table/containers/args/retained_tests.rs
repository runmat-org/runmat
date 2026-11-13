use super::*;

#[test]
fn retained_timetable_partitions_preserve_pair_order_and_original_payload_owners() {
    let text: runmat_value::Utf8Text = "original column payload".into();
    let pointer = text.as_str().as_ptr();
    let input: crate::call::BuiltinArguments = crate::execution::RootedValues::new(vec![
        Value::from("Delimiter"),
        Value::String(runmat_value::StringElement::Text(text)),
        Value::from("RowTimes"),
        Value::Num(1.),
        Value::from("Encoding"),
        Value::from("utf-8"),
        Value::from("RowTimes"),
        Value::Num(2.),
    ])
    .unwrap()
    .into();
    let (table, timetable) =
        futures::executor::block_on(split_readtimetable_options(&input, None)).unwrap();
    assert_eq!(table.len(), 4);
    assert_eq!(table[0], Value::from("Delimiter"));
    assert_eq!(table[2], Value::from("Encoding"));
    assert_eq!(timetable.len(), 4);
    assert_eq!(timetable[1], Value::Num(1.));
    assert_eq!(timetable[3], Value::Num(2.));
    let Value::String(actual) = &table[1] else {
        panic!("retained text")
    };
    assert_eq!(
        actual.as_str().expect("retained nonmissing text").as_ptr(),
        pointer
    );

    let source: crate::call::BuiltinArguments = crate::execution::RootedValues::new(vec![
        Value::Num(3.),
        Value::Num(4.),
        Value::from("RowTimes"),
        Value::Num(5.),
    ])
    .unwrap()
    .into();
    let (times, columns, _) = futures::executor::block_on(
        crate::builtins::table::containers::split_timetable_constructor_args(&source, None),
    )
    .unwrap();
    assert_eq!(times, Some(Value::Num(5.)));
    assert!(columns.iter().eq([Value::Num(3.), Value::Num(4.)].iter()));
}
