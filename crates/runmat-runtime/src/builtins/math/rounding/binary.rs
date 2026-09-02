use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::{ObjectInstance, StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

pub(super) struct TabularBinaryPlan {
    source: ObjectInstance,
    variables: Vec<(String, Value, Value)>,
}

pub(super) enum BinaryInputPlan<T> {
    Values(Box<(Value, Value)>),
    Structured(T),
}

impl TabularBinaryPlan {
    pub(super) fn variables(self) -> (ObjectInstance, Vec<(String, Value, Value)>) {
        (self.source, self.variables)
    }
}

pub(super) fn plan_tabular(
    left: Value,
    right: Value,
) -> Result<BinaryInputPlan<TabularBinaryPlan>, String> {
    let left_table =
        matches!(&left, Value::Object(object) if crate::builtins::table::is_tabular_object(object));
    let right_table = matches!(&right, Value::Object(object) if crate::builtins::table::is_tabular_object(object));
    if !left_table && !right_table {
        return Ok(BinaryInputPlan::Values(Box::new((left, right))));
    }

    match (left, right) {
        (Value::Object(left), Value::Object(right))
            if crate::builtins::table::is_tabular_object(&left)
                && crate::builtins::table::is_tabular_object(&right) =>
        {
            if left.class_name != right.class_name {
                return Err("tabular operands must have the same container class".to_string());
            }
            let left_variables = crate::builtins::table::table_variables(&left)
                .map_err(|error| error.to_string())?;
            let right_variables = crate::builtins::table::table_variables(&right)
                .map_err(|error| error.to_string())?;
            if left_variables
                .field_names()
                .ne(right_variables.field_names())
            {
                return Err(
                    "tabular operands must have matching variable names and order".to_string(),
                );
            }
            let variables = left_variables
                .fields
                .into_iter()
                .zip(right_variables.fields)
                .map(|((name, left), (_, right))| (name, left, right))
                .collect();
            Ok(BinaryInputPlan::Structured(TabularBinaryPlan {
                source: left,
                variables,
            }))
        }
        (Value::Object(source), other) if crate::builtins::table::is_tabular_object(&source) => {
            let variables = crate::builtins::table::table_variables(&source)
                .map_err(|error| error.to_string())?
                .fields
                .into_iter()
                .map(|(name, left)| (name, left, other.clone()))
                .collect();
            Ok(BinaryInputPlan::Structured(TabularBinaryPlan {
                source,
                variables,
            }))
        }
        (other, Value::Object(source)) if crate::builtins::table::is_tabular_object(&source) => {
            let variables = crate::builtins::table::table_variables(&source)
                .map_err(|error| error.to_string())?
                .fields
                .into_iter()
                .map(|(name, right)| (name, other.clone(), right))
                .collect();
            Ok(BinaryInputPlan::Structured(TabularBinaryPlan {
                source,
                variables,
            }))
        }
        (left, right) => Ok(BinaryInputPlan::Values(Box::new((left, right)))),
    }
}

pub(super) fn finish_tabular(
    source: &ObjectInstance,
    variables: Vec<(String, Value)>,
) -> BuiltinResult<Value> {
    let mut output = StructValue::new();
    for (name, value) in variables {
        output.insert(name, value);
    }
    crate::builtins::table::table_replace_variables_like(source, output)
}

pub(super) struct DurationBinaryPlan {
    pub(super) left: Tensor,
    pub(super) right: Tensor,
    format: String,
}

pub(super) fn plan_duration(
    left: Value,
    right: Value,
    builtin: &str,
) -> Result<BinaryInputPlan<DurationBinaryPlan>, String> {
    let left_duration = crate::builtins::duration::is_duration_object(&left);
    let right_duration = crate::builtins::duration::is_duration_object(&right);
    if !left_duration && !right_duration {
        return Ok(BinaryInputPlan::Values(Box::new((left, right))));
    }
    let format = if left_duration {
        crate::builtins::duration::duration_format_from_value(&left)
    } else {
        crate::builtins::duration::duration_format_from_value(&right)
    };
    let left = if left_duration {
        crate::builtins::duration::duration_tensor_from_duration_value(&left)
            .map_err(|error| error.to_string())?
    } else {
        tensor::value_into_tensor_for(builtin, left).map_err(|error| error.to_string())?
    };
    let right = if right_duration {
        crate::builtins::duration::duration_tensor_from_duration_value(&right)
            .map_err(|error| error.to_string())?
    } else {
        tensor::value_into_tensor_for(builtin, right).map_err(|error| error.to_string())?
    };
    Ok(BinaryInputPlan::Structured(DurationBinaryPlan {
        left,
        right,
        format,
    }))
}

pub(super) fn finish_duration(
    plan: DurationBinaryPlan,
    value: Value,
    builtin: &str,
) -> BuiltinResult<Value> {
    let days = tensor::value_into_tensor_for(builtin, value)?;
    crate::builtins::duration::duration_object_from_days_tensor(days, plan.format)
}

pub(super) fn common_resident_owner(
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> Result<&'static dyn AccelProvider, String> {
    let left_owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(left)
        .ok_or_else(|| "no provider owns the left resident operand".to_string())?;
    let right_owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(right)
        .ok_or_else(|| "no provider owns the right resident operand".to_string())?;
    if !std::ptr::eq(left_owner, right_owner) || left.device_id != right.device_id {
        return Err("resident operands must share one owning provider and device".to_string());
    }
    Ok(left_owner)
}

pub(super) fn matching_physical_inputs(left: &GpuTensorHandle, right: &GpuTensorHandle) -> bool {
    left.shape == right.shape
        && runmat_accelerate_api::handle_storage(left)
            == runmat_accelerate_api::handle_storage(right)
        && runmat_accelerate_api::handle_precision(left)
            == runmat_accelerate_api::handle_precision(right)
        && runmat_accelerate_api::handle_integer_type(left)
            == runmat_accelerate_api::handle_integer_type(right)
        && runmat_accelerate_api::handle_is_logical(left)
            == runmat_accelerate_api::handle_is_logical(right)
}

pub(super) fn validate_resident_output(
    provider: &'static dyn AccelProvider,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    output: GpuTensorHandle,
) -> Result<Value, String> {
    let contract = crate::builtins::common::gpu_helpers::BinaryGpuOutputContract {
        shape: left.shape.clone(),
        storage: GpuTensorStorage::Real,
        precision: runmat_accelerate_api::handle_precision(left),
        integer: runmat_accelerate_api::handle_integer_type(left),
        logical: false,
        alias: crate::builtins::common::gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
    };
    if !crate::builtins::common::gpu_helpers::binary_gpu_output_matches(
        &output, left, right, provider, &contract,
    ) {
        crate::builtins::common::gpu_helpers::free_rejected_provider_output(
            &output,
            &[left, right],
            provider,
        );
        return Err("provider returned a malformed binary output".to_string());
    }
    let mut output = output;
    let provenance = if runmat_accelerate_api::handle_is_explicit(left)
        || runmat_accelerate_api::handle_is_explicit(right)
    {
        runmat_accelerate_api::GpuHandleProvenance::Explicit
    } else {
        runmat_accelerate_api::GpuHandleProvenance::Automatic
    };
    runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
    Ok(crate::builtins::common::gpu_helpers::resident_gpu_value(
        output,
    ))
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

    use super::*;
    use crate::builtins::common::test_support;

    fn table(name: &str, values: &[f64]) -> Value {
        crate::builtins::table::table_from_columns(
            vec![name.to_string()],
            vec![Value::Tensor(
                Tensor::new(values.to_vec(), vec![values.len(), 1]).expect("table column"),
            )],
        )
        .expect("table")
    }

    #[test]
    fn tabular_plan_pairs_matching_variables_and_preserves_source_metadata() {
        let mut left = table("A", &[5.0, 8.0]);
        let right = table("A", &[3.0, 4.0]);
        let Value::Object(left_object) = &mut left else {
            panic!("expected table object")
        };
        left_object.class_name = runmat_types::standard::TIMETABLE.owned();
        left_object
            .properties
            .insert("RowTimes".into(), Value::from("preserved"));
        let Value::Object(right_object) = right.clone() else {
            panic!("expected table object")
        };
        let mut right_object = right_object;
        right_object.class_name = runmat_types::standard::TIMETABLE.owned();

        let BinaryInputPlan::Structured(plan) =
            plan_tabular(left, Value::Object(right_object)).expect("matching timetables")
        else {
            panic!("expected tabular plan")
        };
        let (source, variables) = plan.variables();
        assert_eq!(variables.len(), 1);
        assert_eq!(variables[0].0, "A");
        let output = finish_tabular(
            &source,
            vec![(
                "A".into(),
                Value::Tensor(Tensor::new(vec![2.0, 0.0], vec![2, 1]).unwrap()),
            )],
        )
        .expect("finish timetable");
        let Value::Object(output) = output else {
            panic!("expected timetable output")
        };
        assert!(output.is_class(runmat_types::standard::TIMETABLE));
        assert_eq!(
            output.properties.get("RowTimes"),
            Some(&Value::from("preserved"))
        );

        let error = match plan_tabular(table("A", &[1.0]), table("B", &[1.0])) {
            Err(error) => error,
            Ok(_) => panic!("mismatched variable names must reject"),
        };
        assert!(error.contains("matching variable names and order"));
    }

    #[test]
    fn duration_plan_accepts_numeric_day_counts_in_either_operand_position() {
        let duration = crate::builtins::duration::duration_object_from_days_tensor(
            Tensor::new(vec![1.25, 2.5], vec![1, 2]).unwrap(),
            "hh:mm:ss",
        )
        .expect("duration");
        for (left, right) in [
            (duration.clone(), Value::Num(1.0)),
            (Value::Num(1.0), duration.clone()),
        ] {
            let BinaryInputPlan::Structured(plan) =
                plan_duration(left, right, "test remainder").expect("duration plan")
            else {
                panic!("expected duration plan")
            };
            assert_eq!(plan.format, "hh:mm:ss");
            let output = finish_duration(
                plan,
                Value::Tensor(Tensor::new(vec![0.25, 0.5], vec![1, 2]).unwrap()),
                "test remainder",
            )
            .expect("duration output");
            assert!(crate::builtins::duration::is_duration_object(&output));
        }
    }

    #[test]
    fn binary_provider_outputs_require_exact_shape_type_owner_and_non_aliasing() {
        test_support::with_test_provider(|provider| {
            let left = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[-7, 7]),
                    shape: &[2, 1],
                })
                .expect("upload left");
            let right = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[4, -4]),
                    shape: &[2, 1],
                })
                .expect("upload right");
            let valid = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[-3, 3]),
                    shape: &[2, 1],
                })
                .expect("upload valid output");
            let Value::GpuTensor(validated) =
                validate_resident_output(provider, &left, &right, valid)
                    .expect("valid provider output")
            else {
                panic!("validated output must remain resident")
            };
            assert_eq!(validated.shape, vec![2, 1]);

            let mut malformed = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[-3, 3]),
                    shape: &[2, 1],
                })
                .expect("upload malformed output");
            malformed.shape = vec![1, 2];
            let rejected = malformed.clone();
            assert_eq!(
                validate_resident_output(provider, &left, &right, malformed),
                Err("provider returned a malformed binary output".to_string())
            );
            assert!(block_on(provider.download_integer(&rejected)).is_err());

            assert_eq!(
                validate_resident_output(provider, &left, &right, left.clone()),
                Err("provider returned a malformed binary output".to_string())
            );
            assert!(block_on(provider.download_integer(&left)).is_ok());

            for handle in [&left, &right, &validated] {
                provider.free(handle).expect("free binary test handle");
            }
        });
    }
}
