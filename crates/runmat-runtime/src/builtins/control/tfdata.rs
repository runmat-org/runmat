//! Coefficient extraction from the canonical transfer-function model.
use crate::builtins::control::model_polynomial::ss_object_coefficient_grid_async;
use crate::builtins::control::tf_model::{
    coefficient_value, control_error, TfModel, SS_CLASS, TF_CLASS,
};
use crate::{BuiltinResult, RuntimeError};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, ObjectArray, ObjectInstance, Tensor, Value};

const NAME: &str = "tfdata";

fn argument_error(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:tfdata:InvalidArgument", message)
}

#[runtime_builtin(
    name = "tfdata",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::tfdata"
)]
async fn tfdata_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    model: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let vector = rest.len() == 1
        && crate::builtins::common::tensor::value_to_string(&rest[0])
            .is_some_and(|text| text.eq_ignore_ascii_case("v"));
    let requested = crate::output_count::current_output_count().unwrap_or(1);
    if requested > 5 {
        return Err(argument_error("tfdata supports at most five outputs"));
    }
    let model = match model {
        Value::ObjectArray(array) if !rest.is_empty() => {
            if vector {
                return Err(argument_error("'v' requires a scalar SISO model"));
            }
            let index = model_array_index(&array, rest.view()).await?;
            array
                .data()
                .get(index)
                .cloned()
                .ok_or_else(|| argument_error("model array index is out of range"))?
        }
        Value::ObjectArray(array) => {
            return model_array_data(array, requested, runtime.as_ref()).await
        }
        model if rest.is_empty() || vector => model,
        _ => return Err(argument_error("model indices require a model array")),
    };
    let (num, den, sample_time, rows, cols) =
        scalar_model_data(model, vector, runtime.as_ref()).await?;
    publish_coefficients(num, den, sample_time, rows, cols, vector, requested)
}

async fn scalar_model_data(
    model: Value,
    vector: bool,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<(Vec<Value>, Vec<Value>, f64, usize, usize)> {
    match model {
        Value::Object(object) if object.is_class(TF_CLASS) => {
            let model = TfModel::from_value_async(Value::Object(object), NAME).await?;
            let (num, den) = model.coefficient_values(NAME)?;
            Ok((vec![num], vec![den], model.sample_time(), 1, 1))
        }
        Value::Object(object) if object.is_class(SS_CLASS) => {
            // A definitely MIMO model cannot satisfy the vector selector. Do
            // this before downloading resident state matrices: an argument
            // error must not follow a provider effect when host shape facts
            // already determine the answer.
            if vector && definitely_non_siso_ss(&object) {
                return Err(argument_error("'v' requires a SISO model"));
            }
            let (numerators, denominator, rows, cols, sample_time) =
                ss_object_coefficient_grid_async(object, NAME, runtime).await?;
            let num = numerators
                .iter()
                .map(|coefficients| coefficient_value(coefficients, NAME))
                .collect::<BuiltinResult<Vec<_>>>()?;
            let den = (0..numerators.len())
                .map(|_| coefficient_value(&denominator, NAME))
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok((num, den, sample_time, rows, cols))
        }
        _ => Err(control_error(
            NAME,
            "RunMat:tfdata:InvalidModel",
            "tfdata expects a transfer-function or state-space model",
        )),
    }
}

fn publish_coefficients(
    num: Vec<Value>,
    den: Vec<Value>,
    sample_time: f64,
    rows: usize,
    cols: usize,
    vector: bool,
    requested: usize,
) -> BuiltinResult<Value> {
    let (num, den) = if vector {
        if rows != 1 || cols != 1 {
            return Err(argument_error("'v' requires a SISO model"));
        }
        (
            num.into_iter().next().expect("SISO numerator"),
            den.into_iter().next().expect("SISO denominator"),
        )
    } else {
        (
            Value::Cell(
                CellArray::from_column_major(num, vec![rows, cols])
                    .map_err(|error| control_error(NAME, "RunMat:tfdata:Internal", error))?,
            ),
            Value::Cell(
                CellArray::from_column_major(den, vec![rows, cols])
                    .map_err(|error| control_error(NAME, "RunMat:tfdata:Internal", error))?,
            ),
        )
    };
    publish_values(num, den, sample_time, requested)
}

fn publish_values(
    num: Value,
    den: Value,
    sample_time: f64,
    requested: usize,
) -> BuiltinResult<Value> {
    let empty = || -> BuiltinResult<Value> {
        Tensor::new(Vec::new(), vec![0, 0])
            .map(Value::Tensor)
            .map_err(|error| control_error(NAME, "RunMat:tfdata:Internal", error))
    };
    let mut outputs = vec![num, den, Value::Num(sample_time)];
    if requested >= 4 {
        outputs.push(empty()?);
    }
    if requested >= 5 {
        outputs.push(empty()?);
    }
    if requested == 0 {
        Ok(Value::output_list(Vec::new()))
    } else if requested == 1 {
        Ok(outputs.remove(0))
    } else {
        outputs.truncate(requested);
        Ok(Value::output_list(outputs))
    }
}

async fn model_array_index(
    array: &ObjectArray,
    indices: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<usize> {
    let dimensions = array.shape();
    if indices.len() != 1 && indices.len() != dimensions.len() {
        return Err(argument_error(
            "model indexing requires one linear index or one index per array dimension",
        ));
    }
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (axis, value) in indices.iter().enumerate() {
        if matches!(value, Value::GpuTensor(_)) {
            return Err(argument_error("model indices must be host scalar integers"));
        }
        let index = crate::builtins::common::tensor::dimension_from_value_async(
            value,
            "model index",
            false,
        )
        .await
        .map_err(argument_error)?
        .ok_or_else(|| argument_error("model indices must be positive integers"))?;
        let extent = if indices.len() == 1 {
            array.len()
        } else {
            dimensions[axis]
        };
        if index > extent {
            return Err(argument_error("model array index is out of range"));
        }
        offset =
            offset
                .checked_add((index - 1).checked_mul(stride).ok_or_else(|| {
                    argument_error("model array index exceeds addressable storage")
                })?)
                .ok_or_else(|| argument_error("model array index exceeds addressable storage"))?;
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| argument_error("model array size exceeds addressable storage"))?;
    }
    Ok(offset)
}

async fn model_array_data(
    array: ObjectArray,
    requested: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    if array.is_empty() {
        if !array.class_name().is(TF_CLASS) {
            return Err(argument_error(
                "an empty state-space array does not retain input and output counts",
            ));
        }
        if requested > 2 {
            return Err(argument_error(
                "an empty transfer-function array does not retain sample time",
            ));
        }
        let mut cell_shape = vec![1, 1];
        cell_shape.extend_from_slice(array.shape());
        let num = CellArray::from_column_major(Vec::new(), cell_shape.clone())
            .map(Value::Cell)
            .map_err(|error| control_error(NAME, "RunMat:tfdata:Internal", error))?;
        let den = CellArray::from_column_major(Vec::new(), cell_shape)
            .map(Value::Cell)
            .map_err(|error| control_error(NAME, "RunMat:tfdata:Internal", error))?;
        return publish_values(num, den, 0.0, requested);
    }
    let shape = array.shape().to_vec();
    let mut numerators = Vec::new();
    let mut denominators = Vec::new();
    let mut system_extent = None;
    let mut sample_time = None;
    for model in array.data().iter().cloned() {
        let (num, den, ts, rows, cols) = scalar_model_data(model, false, runtime).await?;
        if let Some((expected_rows, expected_cols)) = system_extent {
            if (rows, cols) != (expected_rows, expected_cols) {
                return Err(argument_error(
                    "model array entries must have the same input and output counts",
                ));
            }
        } else {
            system_extent = Some((rows, cols));
        }
        if sample_time.is_some_and(|expected| expected != ts) {
            return Err(argument_error(
                "model array entries must have the same sample time",
            ));
        }
        sample_time = Some(ts);
        numerators.extend(num);
        denominators.extend(den);
    }
    let (rows, cols) = system_extent.expect("nonempty model array");
    let mut cell_shape = vec![rows, cols];
    cell_shape.extend(shape);
    let num = CellArray::from_column_major(numerators, cell_shape.clone())
        .map(Value::Cell)
        .map_err(|error| control_error(NAME, "RunMat:tfdata:Internal", error))?;
    let den = CellArray::from_column_major(denominators, cell_shape)
        .map(Value::Cell)
        .map_err(|error| control_error(NAME, "RunMat:tfdata:Internal", error))?;
    publish_values(
        num,
        den,
        sample_time.expect("nonempty model array"),
        requested,
    )
}

fn definitely_non_siso_ss(object: &ObjectInstance) -> bool {
    let shape = |property: &str| -> Option<&[usize]> {
        match object.properties().get(property)? {
            Value::Tensor(tensor) => Some(tensor.shape()),
            Value::ComplexTensor(tensor) => Some(tensor.shape()),
            Value::GpuTensor(handle) => Some(handle.shape()),
            Value::LogicalArray(array) => Some(&array.shape),
            _ => None,
        }
    };
    let inputs = shape("B").and_then(|shape| match shape {
        [_, columns] => Some(*columns),
        [columns] => Some(*columns),
        _ => None,
    });
    let outputs = shape("C").and_then(|shape| match shape {
        [rows, _] => Some(*rows),
        [_] => Some(1),
        _ => None,
    });
    inputs.is_some_and(|count| count != 1) || outputs.is_some_and(|count| count != 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn vector_selector_preflights_known_non_siso_ports() {
        let mut model = ObjectInstance::new(SS_CLASS.to_owned());
        let matrix = |shape: Vec<usize>| {
            Value::Tensor(Tensor::new(vec![0.0; shape.iter().product()], shape).unwrap())
        };
        model.replace_property_unbounded("B".into(), matrix(vec![2, 1]));
        model.replace_property_unbounded("C".into(), matrix(vec![1, 2]));
        assert!(!definitely_non_siso_ss(&model));

        model.replace_property_unbounded("B".into(), matrix(vec![2, 2]));
        assert!(definitely_non_siso_ss(&model));
        model.replace_property_unbounded("B".into(), matrix(vec![2, 1]));
        model.replace_property_unbounded("C".into(), matrix(vec![2, 2]));
        assert!(definitely_non_siso_ss(&model));
        model.replace_property_unbounded("C".into(), matrix(vec![0, 2]));
        assert!(definitely_non_siso_ss(&model));
        model.replace_property_unbounded("C".into(), matrix(vec![1, 0]));
        model.replace_property_unbounded("B".into(), matrix(vec![0, 0]));
        assert!(definitely_non_siso_ss(&model));
    }

    #[test]
    fn model_array_indices_follow_column_major_order() {
        let objects = (0..4)
            .map(|_| ObjectInstance::new(TF_CLASS.to_owned()))
            .collect();
        let array = ObjectArray::from_objects(TF_CLASS.to_owned(), objects, vec![2, 2]).unwrap();
        assert_eq!(
            block_on(model_array_index(
                &array,
                crate::call::FunctionArgumentView::from_slice(&[Value::Num(3.0)])
            ))
            .unwrap(),
            2
        );
        assert_eq!(
            block_on(model_array_index(
                &array,
                crate::call::FunctionArgumentView::from_slice(&[Value::Num(2.0), Value::Num(1.0)])
            ))
            .unwrap(),
            1
        );
        assert_eq!(
            block_on(model_array_index(
                &array,
                crate::call::FunctionArgumentView::from_slice(&[Value::Num(1.0), Value::Num(2.0)])
            ))
            .unwrap(),
            2
        );
    }

    #[test]
    fn empty_tf_array_retains_coefficient_grid_extent_only() {
        let array = ObjectArray::empty(TF_CLASS.to_owned(), vec![0, 2]).unwrap();
        let Value::Cell(numerators) = block_on(model_array_data(array, 1, None)).unwrap() else {
            panic!("tfdata must return an empty coefficient cell grid")
        };
        assert_eq!(numerators.shape, vec![1, 1, 0, 2]);
        let array = ObjectArray::empty(TF_CLASS.to_owned(), vec![0, 2]).unwrap();
        assert!(block_on(model_array_data(array, 3, None)).is_err());
    }
}
