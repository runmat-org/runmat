use super::capabilities::SECONDS_PER_DAY;
use super::*;

async fn prepare_calendar_unit_input(name: &'static str, value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::CALENDAR_DURATION_GPU_INPUT_EXTENSION,
            name,
        )?;
    }
    gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("{name}: {}", err.message())))
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime",
    context = true
)]
pub(super) async fn calendar_duration_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if !matches!(args.len(), 0 | 1 | 3..=6) {
        return Err(datetime_error(
            "calendarDuration: expected no input, a component matrix, or Y/M/D[/H/M/S] components",
        ));
    }
    if matches!(args.len(), 0 | 4 | 5) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::CALENDAR_DURATION_SHORTHAND_EXTENSION,
            "calendarDuration",
        )?;
    }
    if args.iter().any(|arg| matches!(arg, Value::GpuTensor(_))) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::CALENDAR_DURATION_GPU_INPUT_EXTENSION,
            "calendarDuration",
        )?;
    }
    if args
        .iter()
        .any(|arg| matches!(arg, Value::Bool(_) | Value::LogicalArray(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::CALENDAR_DURATION_LOGICAL_COMPONENT_EXTENSION,
            "calendarDuration",
        )?;
    }
    if args.len() == 1 && is_calendar_duration_object(&args[0]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::CALENDAR_DURATION_SHORTHAND_EXTENSION,
            "calendarDuration",
        )?;
    }
    let args = gather_retained_args(args, runtime.as_ref()).await?;
    if args.is_empty() {
        return calendar_duration_object_from_components(
            runtime.as_ref(),
            vec![0.0],
            vec![0.0],
            vec![1, 1],
        );
    }
    if args.len() == 1 && is_calendar_duration_object(&args[0]) {
        return Ok(args[0].clone());
    }

    let labels = ["years", "months", "days", "hours", "minutes", "seconds"];
    let positional = match args.len() {
        1 => {
            let components = component_tensor(args[0].clone(), "calendarDuration")?;
            let shape = components.shape().to_vec();
            if shape.len() == 2 && matches!(shape[1], 3 | 6) {
                let rows = shape[0];
                let columns = shape[1];
                let values = tensor::tensor_into_values_f64(components);
                let mut months = Vec::with_capacity(rows);
                let mut days = Vec::with_capacity(rows);
                for row in 0..rows {
                    let part = |column: usize| values[row + column * rows];
                    for column in 0..columns {
                        validate_calendar_component(part(column), column)?;
                    }
                    let month = part(0) * 12.0 + part(1);
                    let day = part(2)
                        + if columns == 6 {
                            part(3) / 24.0 + part(4) / (24.0 * 60.0) + part(5) / SECONDS_PER_DAY
                        } else {
                            0.0
                        };
                    if !month.is_finite() || !day.is_finite() {
                        return Err(datetime_error(
                            "calendarDuration: resulting calendar duration is outside supported range",
                        ));
                    }
                    months.push(month);
                    days.push(day);
                }
                return calendar_duration_object_from_components(
                    runtime.as_ref(),
                    months,
                    days,
                    vec![rows, 1],
                );
            }
            crate::compatibility::ensure_builtin_extension_enabled(
                &runmat_builtins::CALENDAR_DURATION_SHORTHAND_EXTENSION,
                "calendarDuration",
            )?;
            let day_values = tensor::tensor_into_values_f64(components);
            for value in day_values.iter().copied() {
                validate_calendar_component(value, 2)?;
            }
            return calendar_duration_object_from_components(
                runtime.as_ref(),
                vec![0.0; day_values.len()],
                day_values,
                shape,
            );
        }
        3 | 6 => args,
        4 | 5 => args,
        _ => {
            return Err(datetime_error(
                "calendarDuration: expected no input, days, or Y/M/D[/H/M/S] components",
            ))
        }
    };

    let mut arrays = Vec::with_capacity(6);
    for (idx, arg) in positional.into_iter().enumerate() {
        arrays.push(component_tensor(arg, labels[idx])?);
    }
    while arrays.len() < 6 {
        arrays.push(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
    }
    let (broadcasted, shape) = broadcast_component_data(&arrays, &labels)?;
    let len = broadcasted[0].len();
    let mut months = Vec::with_capacity(len);
    let mut days = Vec::with_capacity(len);
    for idx in 0..len {
        for (component, values) in broadcasted.iter().enumerate() {
            validate_calendar_component(values[idx], component)?;
        }
        let month_value = broadcasted[0][idx] * 12.0 + broadcasted[1][idx];
        let day_value = broadcasted[2][idx]
            + broadcasted[3][idx] / 24.0
            + broadcasted[4][idx] / (24.0 * 60.0)
            + broadcasted[5][idx] / SECONDS_PER_DAY;
        if !month_value.is_finite() || !day_value.is_finite() {
            return Err(datetime_error(
                "calendarDuration: resulting calendar duration is outside supported range",
            ));
        }
        months.push(month_value);
        days.push(day_value);
    }
    calendar_duration_object_from_components(runtime.as_ref(), months, days, shape)
}

fn validate_calendar_component(value: f64, component: usize) -> BuiltinResult<()> {
    if !value.is_finite() || (component != 5 && value.fract() != 0.0) {
        return Err(datetime_error(
            "calendarDuration: calendar components must be finite; years, months, days, hours and minutes must be integers",
        ));
    }
    Ok(())
}

#[runmat_macros::runtime_builtin(
    name = "caldays",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn caldays_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = prepare_calendar_unit_input("caldays", value).await?;
    calendar_duration_unit_value(value, "caldays", 0.0, 1.0)
}

#[runmat_macros::runtime_builtin(
    name = "calweeks",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calweeks_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = prepare_calendar_unit_input("calweeks", value).await?;
    calendar_duration_unit_value(value, "calweeks", 0.0, 7.0)
}

#[runmat_macros::runtime_builtin(
    name = "calmonths",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calmonths_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = prepare_calendar_unit_input("calmonths", value).await?;
    calendar_duration_unit_value(value, "calmonths", 1.0, 0.0)
}

#[runmat_macros::runtime_builtin(
    name = "calquarters",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calquarters_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = prepare_calendar_unit_input("calquarters", value).await?;
    calendar_duration_unit_value(value, "calquarters", 3.0, 0.0)
}

#[runmat_macros::runtime_builtin(
    name = "calyears",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn calyears_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = prepare_calendar_unit_input("calyears", value).await?;
    calendar_duration_unit_value(value, "calyears", 12.0, 0.0)
}

#[runmat_macros::runtime_builtin(
    name = "iscalendarduration",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) fn iscalendarduration_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(is_calendar_duration_object(&value)))
}

#[cfg(test)]
pub(crate) async fn calendar_duration_builtin_source(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let retained = crate::call::BuiltinArguments::from_borrowed(&args, runtime.as_ref())?;
    calendar_duration_builtin(runtime, retained).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_calendar_unit_helpers_accept_logical_arrays_in_both_modes() {
        let logical =
            Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap());
        for enabled in [false, true] {
            let _mode = crate::compatibility::push_runmat_extensions_enabled(enabled);
            let results = [
                futures::executor::block_on(caldays_builtin(logical.clone())),
                futures::executor::block_on(calweeks_builtin(logical.clone())),
                futures::executor::block_on(calmonths_builtin(logical.clone())),
                futures::executor::block_on(calquarters_builtin(logical.clone())),
                futures::executor::block_on(calyears_builtin(logical.clone())),
            ];
            for result in results {
                let (months, days) =
                    calendar_duration_tensors_from_value(&result.unwrap()).unwrap();
                assert_eq!(months.shape(), [1, 2]);
                assert_eq!(days.shape(), [1, 2]);
                assert_eq!(
                    (months.materialize_f64()[1], days.materialize_f64()[1]),
                    (0.0, 0.0)
                );
            }
        }
    }

    #[test]
    fn unit_helpers_keep_months_and_days_separate_and_reject_fractions() {
        let month = futures::executor::block_on(calmonths_builtin(Value::Num(2.0))).unwrap();
        let day = futures::executor::block_on(caldays_builtin(Value::Num(5.0))).unwrap();
        assert_eq!(
            futures::executor::block_on(calmonths_builtin(month.clone())).unwrap(),
            Value::Num(2.0)
        );
        assert_eq!(
            futures::executor::block_on(caldays_builtin(month)).unwrap(),
            Value::Num(0.0)
        );
        assert_eq!(
            futures::executor::block_on(calmonths_builtin(day.clone())).unwrap(),
            Value::Num(0.0)
        );
        assert_eq!(
            futures::executor::block_on(caldays_builtin(day)).unwrap(),
            Value::Num(5.0)
        );
        for fractional in [
            Value::Num(1.5),
            Value::Num(f64::INFINITY),
            Value::Num(f64::NAN),
        ] {
            assert!(futures::executor::block_on(calweeks_builtin(fractional)).is_err());
        }
        let integer = futures::executor::block_on(calmonths_builtin(Value::Int(
            runmat_value::IntValue::I16(3),
        )))
        .unwrap();
        assert_eq!(
            futures::executor::block_on(calmonths_builtin(integer)).unwrap(),
            Value::Num(3.0)
        );
        assert!(futures::executor::block_on(calyears_builtin(Value::Int(
            runmat_value::IntValue::U64(u64::MAX),
        )))
        .is_err());
    }

    #[test]
    fn matrix_constructor_uses_column_major_rows_and_validates_components() {
        let matrix = Tensor::new(vec![1.0, 0.0, 2.0, 4.0, 3.0, 5.0], vec![2, 3]).unwrap();
        let value = futures::executor::block_on(calendar_duration_builtin_source(
            None,
            vec![Value::Tensor(matrix)],
        ))
        .unwrap();
        let (months, days) = calendar_duration_tensors_from_value(&value).unwrap();
        assert_eq!(months.shape(), [2, 1]);
        assert_eq!(months.materialize_f64(), vec![14.0, 4.0]);
        assert_eq!(days.materialize_f64(), vec![3.0, 5.0]);
        assert_eq!(
            iscalendarduration_builtin(value).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            iscalendarduration_builtin(Value::Num(1.0)).unwrap(),
            Value::Bool(false)
        );
        let fractional = Tensor::new(vec![1.0, 0.0, 2.5], vec![1, 3]).unwrap();
        assert!(
            futures::executor::block_on(calendar_duration_builtin_source(
                None,
                vec![Value::Tensor(fractional)]
            ))
            .is_err()
        );
        let with_seconds = Tensor::new(vec![0.0, 0.0, 1.0, 1.0, 30.0, 0.5], vec![1, 6]).unwrap();
        let value = futures::executor::block_on(calendar_duration_builtin_source(
            None,
            vec![Value::Tensor(with_seconds)],
        ))
        .unwrap();
        let (_, days) = calendar_duration_tensors_from_value(&value).unwrap();
        let expected = 1.0 + 1.0 / 24.0 + 30.0 / 1440.0 + 0.5 / SECONDS_PER_DAY;
        assert!((days.materialize_f64()[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn empty_component_shapes_and_matrix_rows_are_preserved() {
        let empty_matrix = Value::Tensor(Tensor::new(vec![], vec![0, 3]).unwrap());
        let constructed = futures::executor::block_on(calendar_duration_builtin_source(
            None,
            vec![empty_matrix.clone()],
        ))
        .unwrap();
        let (months, days) = calendar_duration_tensors_from_value(&constructed).unwrap();
        assert_eq!(months.shape(), [0, 1]);
        assert_eq!(days.shape(), [0, 1]);

        let unit = futures::executor::block_on(caldays_builtin(empty_matrix)).unwrap();
        let (months, days) = calendar_duration_tensors_from_value(&unit).unwrap();
        assert_eq!(months.shape(), [0, 3]);
        assert_eq!(days.shape(), [0, 3]);
        let Value::Tensor(extracted) = futures::executor::block_on(caldays_builtin(unit)).unwrap()
        else {
            panic!("expected empty numeric tensor")
        };
        assert_eq!(extracted.shape(), [0, 3]);

        let empty_component = Value::Tensor(Tensor::new(vec![], vec![0, 2]).unwrap());
        let value = futures::executor::block_on(calendar_duration_builtin_source(
            None,
            vec![empty_component.clone(), Value::Num(1.0), Value::Num(2.0)],
        ))
        .unwrap();
        assert_eq!(
            calendar_duration_tensors_from_value(&value)
                .unwrap()
                .0
                .shape(),
            [0, 2]
        );
        assert!(
            futures::executor::block_on(calendar_duration_builtin_source(
                None,
                vec![
                    empty_component,
                    Value::Tensor(Tensor::new(vec![], vec![0, 3]).unwrap()),
                    Value::Num(2.0)
                ]
            ))
            .is_err()
        );
    }

    #[test]
    fn matlab_mode_rejects_runmat_constructor_shortcuts() {
        let _mode = crate::compatibility::push_runmat_extensions_enabled(false);
        assert!(
            futures::executor::block_on(calendar_duration_builtin_source(None, vec![])).is_err()
        );
        assert!(
            futures::executor::block_on(calendar_duration_builtin_source(
                None,
                vec![Value::Num(2.0)]
            ))
            .is_err()
        );
        assert!(
            futures::executor::block_on(calendar_duration_builtin_source(
                None,
                vec![Value::Bool(true), Value::Num(2.0), Value::Num(3.0)]
            ))
            .is_err()
        );
        let standard = futures::executor::block_on(calendar_duration_builtin_source(
            None,
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
        ))
        .unwrap();
        assert!(
            futures::executor::block_on(calendar_duration_builtin_source(
                None,
                vec![standard.clone()]
            ))
            .is_err()
        );
        for count in [4, 5] {
            assert!(
                futures::executor::block_on(calendar_duration_builtin_source(
                    None,
                    vec![Value::Num(1.0); count]
                ))
                .is_err()
            );
        }
        drop(_mode);
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for count in [4, 5] {
            assert!(
                futures::executor::block_on(calendar_duration_builtin_source(
                    None,
                    vec![Value::Num(1.0); count]
                ))
                .is_ok()
            );
        }
        assert!(
            futures::executor::block_on(calendar_duration_builtin_source(None, vec![standard]))
                .is_ok()
        );
    }

    #[test]
    fn resident_inputs_gate_before_access_and_use_the_owner_in_runmat_mode() {
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let resident = Value::GpuTensor(allocation.handle.clone());
        {
            let _mode = crate::compatibility::push_runmat_extensions_enabled(false);
            let attempts = [
                futures::executor::block_on(caldays_builtin(resident.clone())),
                futures::executor::block_on(calweeks_builtin(resident.clone())),
                futures::executor::block_on(calmonths_builtin(resident.clone())),
                futures::executor::block_on(calquarters_builtin(resident.clone())),
                futures::executor::block_on(calyears_builtin(resident.clone())),
                futures::executor::block_on(calendar_duration_builtin_source(
                    None,
                    vec![resident.clone()],
                )),
            ];
            for result in attempts {
                assert_eq!(
                    result.unwrap_err().identifier(),
                    Some("RunMat:compatibility:DatetimeGpuInputExtension")
                );
            }
            assert_eq!(
                iscalendarduration_builtin(resident).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(allocation.probe.retirements(), 0);
        }
        allocation.handle.release().unwrap();

        crate::builtins::common::test_support::with_test_provider(|provider| {
            let values = [2.0, 3.0];
            let shape = [1, 2];
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .unwrap();
            let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
            let result =
                futures::executor::block_on(calmonths_builtin(Value::GpuTensor(handle))).unwrap();
            let (months, days) = calendar_duration_tensors_from_value(&result).unwrap();
            assert_eq!(months.materialize_f64(), vec![2.0, 3.0]);
            assert_eq!(days.materialize_f64(), vec![0.0, 0.0]);
        });
    }
}
