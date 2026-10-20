use super::*;
async fn table2array_builtin(value: Value) -> BuiltinResult<Value> {
    super::table2array_builtin(None, value).await
}
async fn table2cell_builtin(value: Value) -> BuiltinResult<Value> {
    super::table2cell_builtin(None, value).await
}

async fn cell2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::cell2table_builtin(None, value, rest).await
}

async fn struct2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::struct2table_builtin(None, value, rest).await
}

async fn table2struct_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::table2struct_builtin(None, value, rest).await
}

use runmat_builtins::{BuiltinCompletionPolicy, BuiltinOutputMode, CELL2TABLE_DESCRIPTOR};

async fn array2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::array2table_builtin(None, value, rest).await
}

#[test]
fn table2array_of_cell_variables_matches_element_cells_in_row_major_layout() {
    let table = table_from_columns(
        vec!["A".into(), "B".into()],
        vec![
            Value::Cell(CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 2, 1).unwrap()),
            Value::Cell(CellArray::new(vec![Value::Num(3.0), Value::Num(4.0)], 2, 1).unwrap()),
        ],
    )
    .unwrap();
    let expected = futures::executor::block_on(table2cell_builtin(table.clone())).unwrap();
    let result = futures::executor::block_on(table2array_builtin(table)).unwrap();
    assert_eq!(result, expected);
    let Value::Cell(cells) = result else {
        panic!("expected element cells")
    };
    assert_eq!((cells.rows, cells.cols), (2, 2));
    assert_eq!(cells.get(0, 0).unwrap(), Value::Num(1.0));
    assert_eq!(cells.get(1, 0).unwrap(), Value::Num(2.0));
    assert_eq!(cells.get(0, 1).unwrap(), Value::Num(3.0));
    assert_eq!(cells.get(1, 1).unwrap(), Value::Num(4.0));
}

#[test]
fn complex_single_columns_and_row_elements_keep_single_storage() {
    let input = runmat_value::ComplexTensor::from_f32(
        vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)],
        vec![2, 2],
    )
    .unwrap();
    let table =
        futures::executor::block_on(array2table_builtin(Value::ComplexTensor(input), Vec::new()))
            .unwrap();
    let variables = table_variables(&into_table_object(table.clone(), "test").unwrap()).unwrap();
    for column in variables.fields.values() {
        let Value::ComplexTensor(column) = column else {
            panic!("expected complex column")
        };
        assert_eq!(column.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(column.shape(), &[2, 1]);
    }
    let Value::Cell(cells) =
        futures::executor::block_on(table2cell_builtin(table.clone())).unwrap()
    else {
        panic!("expected element cells")
    };
    let Value::ComplexTensor(element) = cells.get(0, 1).unwrap() else {
        panic!("expected single complex element")
    };
    assert_eq!(element.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(element.materialize_f64(), vec![(3.0, -3.0)]);
    let Value::StructArray(rows) =
        futures::executor::block_on(table2struct_builtin(table, Vec::new())).unwrap()
    else {
        panic!("expected row structures")
    };
    let Value::ComplexTensor(element) = rows.get_linear(1).unwrap().fields.get("Var1").unwrap()
    else {
        panic!("expected single complex field")
    };
    assert_eq!(element.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(element.materialize_f64(), vec![(2.0, -2.0)]);
}

#[test]
fn complex_wide_integer_row_elements_remain_exact() {
    let input = runmat_value::ComplexTensor::new_integer(
        runmat_value::IntegerComplexStorage::new(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            runmat_value::IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
        )
        .unwrap(),
        vec![2, 1],
    )
    .unwrap();
    let table =
        futures::executor::block_on(array2table_builtin(Value::ComplexTensor(input), Vec::new()))
            .unwrap();
    let Value::Cell(cells) =
        futures::executor::block_on(table2cell_builtin(table.clone())).unwrap()
    else {
        panic!("expected element cells")
    };
    let Value::ComplexTensor(first) = cells.get(0, 0).unwrap() else {
        panic!("expected integer complex element")
    };
    let storage = first.integer_storage().unwrap();
    assert_eq!(
        storage.real(),
        &runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993])
    );
    assert_eq!(
        storage.imag(),
        &runmat_value::IntegerStorage::U64(vec![u64::MAX])
    );
    let Value::StructArray(rows) =
        futures::executor::block_on(table2struct_builtin(table, Vec::new())).unwrap()
    else {
        panic!("expected row structures")
    };
    let Value::ComplexTensor(second) = rows.get_linear(1).unwrap().fields.get("Var1").unwrap()
    else {
        panic!("expected integer complex field")
    };
    let storage = second.integer_storage().unwrap();
    assert_eq!(
        storage.real(),
        &runmat_value::IntegerStorage::U64(vec![u64::MAX])
    );
    assert_eq!(
        storage.imag(),
        &runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993])
    );
}

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
        let mut first = runmat_value::UnboundedStructValueBuilder::new();
        first.insert("id", Value::Int(IntValue::U64(u64::MAX)));
        let mut second = runmat_value::UnboundedStructValueBuilder::new();
        second.insert("id", Value::Int(IntValue::U64(1_u64 << 63)));
        let second = second.finish();
        let first = first.finish();
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
