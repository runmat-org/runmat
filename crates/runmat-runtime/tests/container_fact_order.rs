use futures::executor::block_on;
use runmat_builtins::{infer_catalog_call, STRUCT_CATALOG_ENTRY};
use runmat_runtime::{call_builtin_async_with_outputs, value_fact::value_fact};
use runmat_types::{CallRequest, LiteralContext, LiteralValue, OutputSelection, ValueKindFact};
use runmat_value::{CellArray, IntValue, Value};

#[test]
fn runtime_and_catalog_struct_expansion_agree_on_cell_linear_order() {
    for shape in [vec![2, 2], vec![2, 1, 2]] {
        let linear = vec![
            Value::Num(1.0),
            Value::Bool(true),
            Value::String("third".into()),
            Value::Int(IntValue::U64(u64::MAX)),
        ];
        let cells = Value::Cell(CellArray::from_column_major(linear.clone(), shape).unwrap());
        let cell_fact = value_fact(&cells);
        let ValueKindFact::Cell(contents) = &cell_fact.kind else {
            panic!("cell fact")
        };
        assert_eq!(
            contents.elements,
            linear.iter().map(value_fact).collect::<Vec<_>>()
        );
        let args = vec![Value::String("payload".into()), cells];
        let expected = infer_catalog_call(
            &STRUCT_CATALOG_ENTRY,
            &CallRequest {
                arguments: args.iter().map(value_fact).collect(),
                literals: LiteralContext::new(vec![LiteralValue::String("payload".into())]),
                outputs: OutputSelection::default(),
            },
        );
        assert!(expected.diagnostics.is_empty());
        let result = block_on(call_builtin_async_with_outputs("struct", &args, 1)).unwrap();
        let actual = value_fact(&result);
        assert_eq!(expected.outputs[0].shape, actual.shape);
        let (ValueKindFact::Struct(expected), ValueKindFact::Struct(actual)) =
            (&expected.outputs[0].kind, &actual.kind)
        else {
            panic!("struct facts")
        };
        assert_eq!(expected.elements, actual.elements);
        assert!(actual.elements_complete);
    }
}
