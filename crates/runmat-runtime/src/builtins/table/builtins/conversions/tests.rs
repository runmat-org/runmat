use super::*;
use runmat_builtins::{BuiltinCompletionPolicy, BuiltinOutputMode, CELL2TABLE_DESCRIPTOR};

#[cfg(test)]
mod cell2table_tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    #[test]
    fn dedicated_descriptor_is_fixed_and_public() {
        assert_eq!(CELL2TABLE_DESCRIPTOR.signatures.len(), 1);
        assert!(CELL2TABLE_DESCRIPTOR.signatures[0]
            .label
            .starts_with("T = cell2table"));
        assert!(matches!(
            CELL2TABLE_DESCRIPTOR.output_mode,
            BuiltinOutputMode::Fixed
        ));
        assert!(matches!(
            CELL2TABLE_DESCRIPTOR.completion_policy,
            BuiltinCompletionPolicy::Public
        ));
    }

    #[test]
    fn compatible_integer_columns_preserve_and_saturate_leftmost_class() {
        let exact = vec![
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
            ),
            Value::Int(IntValue::U64(1_u64 << 63)),
        ];
        let Value::Tensor(exact) = compatible_integer_column(&exact, 2).unwrap().unwrap() else {
            panic!("expected integer variable");
        };
        assert_eq!(
            exact.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
        );

        let mixed = vec![
            Value::Int(IntValue::U8(9)),
            Value::Int(IntValue::I64(-1)),
            Value::Int(IntValue::U64(u64::MAX)),
        ];
        let Value::Tensor(mixed) = compatible_integer_column(&mixed, 3).unwrap().unwrap() else {
            panic!("expected integer variable");
        };
        assert_eq!(
            mixed.integer_storage(),
            Some(&IntegerStorage::U8(vec![9, 0, u8::MAX]))
        );

        let scalar_double = vec![Value::Num(300.2), Value::Int(IntValue::U8(7))];
        let Value::Tensor(scalar_double) = compatible_integer_column(&scalar_double, 2)
            .unwrap()
            .unwrap()
        else {
            panic!("expected integer variable");
        };
        assert_eq!(
            scalar_double.integer_storage(),
            Some(&IntegerStorage::U8(vec![u8::MAX, 7]))
        );

        assert!(
            compatible_integer_column(&[Value::Int(IntValue::U64(1)), Value::Num(2.0)], 2)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn compatible_integer_columns_preserve_every_native_class() {
        for value in [
            IntValue::I8(7),
            IntValue::I16(7),
            IntValue::I32(7),
            IntValue::I64(7),
            IntValue::U8(7),
            IntValue::U16(7),
            IntValue::U32(7),
            IntValue::U64(7),
        ] {
            let expected_dtype = integer_scalar_prototype(&value).numeric_dtype();
            let Value::Tensor(column) =
                compatible_integer_column(&[Value::Int(value.clone()), Value::Int(value)], 2)
                    .unwrap()
                    .unwrap()
            else {
                panic!("expected integer variable");
            };
            assert_eq!(column.numeric_dtype(), expected_dtype);
            assert_eq!(column.materialize_f64(), vec![7.0, 7.0]);
        }
    }

    #[test]
    fn incompatible_columns_remain_cells_and_rank_over_two_rejects() {
        assert!(
            compatible_integer_column(&[Value::Int(IntValue::U8(1)), Value::from("text")], 2)
                .unwrap()
                .is_none()
        );

        let cell =
            CellArray::new_with_shape(vec![Value::Int(IntValue::U8(1))], vec![1, 1, 1]).unwrap();
        let err = block_on(cell2table_builtin(Value::Cell(cell), Vec::new())).unwrap_err();
        assert!(err.message().contains("two-dimensional"));
    }

    #[test]
    fn struct_array_integer_fields_become_exact_homogeneous_variables() {
        let mut first = StructValue::new();
        first.insert("id", Value::Int(IntValue::U64(u64::MAX)));
        let mut second = StructValue::new();
        second.insert("id", Value::Int(IntValue::U64(1_u64 << 63)));
        let structs = runmat_value::StructArray::with_fields(
            vec!["id".into()],
            vec![first.clone(), second.clone()],
            vec![2, 1],
        )
        .expect("structure array");

        let table = block_on(struct2table_builtin(
            Value::StructArray(structs),
            Vec::new(),
        ))
        .expect("struct2table");
        let variables = table_variables(&into_table_object(table, "test").unwrap()).unwrap();
        let Value::Tensor(ids) = variables.fields.get("id").expect("id variable") else {
            panic!("expected homogeneous integer variable");
        };
        assert_eq!(
            ids.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
        );

        let cell = CellArray::new(vec![Value::Struct(first), Value::Struct(second)], 2, 1)
            .expect("ordinary cell array");
        let error = block_on(struct2table_builtin(Value::Cell(cell), Vec::new()))
            .expect_err("cell contents do not define structure-array identity");
        assert!(error.message().contains("expected struct or struct array"));
    }

    #[test]
    fn struct2table_as_array_requires_numeric_zero_or_one() {
        let err = block_on(struct2table_builtin(
            Value::Struct(StructValue::new()),
            vec![Value::from("AsArray"), Value::Int(IntValue::U8(2))],
        ))
        .expect_err("AsArray=2 must reject");
        assert!(err.message().contains("0 or 1"));
    }
}
