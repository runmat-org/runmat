use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::{ObjectInstance, StructValue, Value};

use crate::BuiltinResult;

pub(crate) enum BinaryInputPlan<T> {
    Values(Box<(Value, Value)>),
    Structured(T),
}

pub(crate) struct TabularBinaryPlan {
    source: ObjectInstance,
    variables: Vec<(String, Value, Value)>,
}

impl TabularBinaryPlan {
    pub(crate) fn variables(self) -> (ObjectInstance, Vec<(String, Value, Value)>) {
        (self.source, self.variables)
    }
}

pub(crate) fn plan_tabular(
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

pub(crate) fn finish_tabular(
    source: &ObjectInstance,
    variables: Vec<(String, Value)>,
) -> BuiltinResult<Value> {
    let mut output = StructValue::new();
    for (name, value) in variables {
        output.insert(name, value);
    }
    crate::builtins::table::table_replace_variables_like(source, output)
}

pub(crate) fn matching_physical_inputs(left: &GpuTensorHandle, right: &GpuTensorHandle) -> bool {
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

pub(crate) fn validate_resident_output(
    provider: &'static dyn AccelProvider,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    output: GpuTensorHandle,
    contract: &super::gpu_helpers::BinaryGpuOutputContract,
) -> Result<Value, String> {
    if !super::gpu_helpers::binary_gpu_output_matches(&output, left, right, provider, contract) {
        super::gpu_helpers::free_rejected_provider_output(&output, &[left, right], provider);
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
    Ok(super::gpu_helpers::resident_gpu_value(output))
}

#[cfg(test)]
mod tests {
    use runmat_value::Tensor;

    use super::*;

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
        let Value::Object(right_object) = right else {
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
}
