use super::*;
use std::sync::Arc;

#[test]
fn asymmetric_uniform_and_cell_multioutputs_keep_linear_positions() {
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_, arguments, count| {
            assert_eq!(count, 2);
            let Value::Num(value) = arguments[0] else {
                panic!("numeric input")
            };
            Box::pin(async move {
                crate::sequence::ValueSequence::comma_separated(vec![
                    Value::Num(value),
                    Value::Num(value + 10.0),
                ])
                .map_err(crate::sequence::sequence_error_to_runtime)
            })
        },
    )));
    for shape in [vec![2, 2], vec![2, 2, 2]] {
        let len = shape.iter().product::<usize>();
        let linear = (1..=len).map(|n| Value::Num(n as f64)).collect::<Vec<_>>();
        for uniform in [true, false] {
            let _outputs = crate::output_count::push_output_count(Some(2));
            let result = call(
                Value::BoundFunctionHandle {
                    name: "position".into(),
                    function: 821,
                },
                vec![
                    Value::Cell(
                        CellArray::from_column_major(linear.clone(), shape.clone()).unwrap(),
                    ),
                    Value::String("UniformOutput".into()),
                    Value::Bool(uniform),
                ],
            )
            .unwrap();
            let Value::OutputList(outputs) = result else {
                panic!("two outputs")
            };
            for (slot, output) in outputs.into_iter().enumerate() {
                let expected = (1..=len)
                    .map(|n| n as f64 + 10.0 * slot as f64)
                    .collect::<Vec<_>>();
                match output {
                    Value::Tensor(tensor) if uniform => {
                        assert_eq!(tensor.shape, shape);
                        assert_eq!(tensor.materialize_f64(), expected);
                    }
                    Value::Cell(cell) if !uniform => {
                        assert_eq!(cell.shape, shape);
                        assert_eq!(
                            cell.to_column_major(),
                            expected.into_iter().map(Value::Num).collect::<Vec<_>>()
                        );
                    }
                    other => panic!("unexpected output {other:?}"),
                }
            }
        }
    }
}

#[test]
fn handler_context_refers_to_the_original_column_major_element() {
    for shape in [vec![2, 2], vec![2, 2, 2]] {
        let expected_shape = shape.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_, arguments, _| {
                let [Value::Struct(context), Value::Num(value)] = arguments else {
                    panic!("handler arguments")
                };
                assert_eq!(context.fields["index"], Value::Num(*value));
                let Value::Tensor(indices) = &context.fields["indices"] else {
                    panic!("indices")
                };
                let mut linear = *value as usize - 1;
                let expected = expected_shape
                    .iter()
                    .map(|dimension| {
                        let coordinate = (linear % dimension + 1) as f64;
                        linear /= dimension;
                        coordinate
                    })
                    .collect::<Vec<_>>();
                assert_eq!(indices.materialize_f64(), expected);
                let value = *value;
                crate::sequence::single_value_future(async move { Ok(Value::Num(value)) })
            },
        )));
        let linear = (1..=shape.iter().product::<usize>())
            .map(|n| Value::Num(n as f64))
            .collect::<Vec<_>>();
        let result = call(
            Value::FunctionHandle("missing_linear_callback".into()),
            vec![
                Value::Cell(CellArray::from_column_major(linear.clone(), shape).unwrap()),
                Value::String("ErrorHandler".into()),
                Value::BoundFunctionHandle {
                    name: "position_handler".into(),
                    function: 822,
                },
            ],
        )
        .unwrap();
        assert_eq!(
            tensor_values(result),
            linear
                .into_iter()
                .map(|v| match v {
                    Value::Num(n) => n,
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>()
        );
    }
}
