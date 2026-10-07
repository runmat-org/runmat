use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult};
use runmat_value::{ComplexTensor, LogicalArray, Tensor, Value};

/// Recover an elementwise singleton's shape after a host evaluator used a scalar
/// carrier or a canonical 1-by-1 typed carrier. Only elementwise owners may call
/// this: it does not infer a reduction shape or reshape a non-scalar result.
/// Class and complexity remain unchanged.
pub(in crate::builtins::math) fn preserve_elementwise_scalar_shape(
    output: Value,
    input_shape: &[usize],
    builtin: &str,
) -> BuiltinResult<Value> {
    let canonical_scalar = match &output {
        Value::Num(_) | Value::Complex(_, _) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor.shape() == [1, 1],
        Value::ComplexTensor(tensor) => tensor.shape() == [1, 1],
        Value::LogicalArray(array) => array.shape == [1, 1],
        _ => false,
    };
    if !canonical_scalar {
        return Ok(output);
    }
    let error = |detail| {
        build_runtime_error(format!("{builtin}: {detail}"))
            .with_builtin(builtin)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    };
    // Empty dimensions are not scalar; all-one axes are cardinality one
    // without any product overflow. The empty shape is the scalar descriptor.
    if input_shape.iter().any(|&extent| extent != 1) {
        return Err(error(
            "elementwise host fallback returned a scalar for nonscalar input".to_string(),
        ));
    }
    match output {
        Value::Complex(real, imaginary) => {
            ComplexTensor::new(vec![(real, imaginary)], input_shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(error)
        }
        Value::ComplexTensor(tensor) => {
            ComplexTensor::from_complex_storage(tensor.into_complex_storage(), input_shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(error)
        }
        Value::LogicalArray(array) => {
            LogicalArray::from_host_buffer(array.data, input_shape.to_vec())
                .map(Value::LogicalArray)
                .map_err(error)
        }
        Value::Bool(value) => LogicalArray::new(vec![u8::from(value)], input_shape.to_vec())
            .map(Value::LogicalArray)
            .map_err(error),
        scalar => {
            let tensor = tensor::value_into_tensor_for(builtin, scalar).map_err(error)?;
            let storage = tensor.into_numeric_storage().map_err(error)?;
            Tensor::from_numeric_storage(storage, input_shape.to_vec())
                .map(Value::Tensor)
                .map_err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntValue, IntegerStorage};

    #[test]
    fn elementwise_scalar_shape_preserves_class_and_rejects_wrong_counts() {
        let Value::Tensor(integer) =
            preserve_elementwise_scalar_shape(Value::Int(IntValue::I16(7)), &[1, 1, 1], "fixture")
                .unwrap()
        else {
            panic!("typed tensor expected");
        };
        assert_eq!(integer.shape(), &[1, 1, 1]);
        assert_eq!(
            integer.integer_storage(),
            Some(&IntegerStorage::I16(vec![7]))
        );
        let Value::ComplexTensor(complex) =
            preserve_elementwise_scalar_shape(Value::Complex(0.0, 2.0), &[1, 1, 1], "fixture")
                .unwrap()
        else {
            panic!("complex tensor expected");
        };
        assert_eq!(complex.shape(), &[1, 1, 1]);
        assert_eq!(complex.materialize_f64(), vec![(0.0, 2.0)]);
        let Value::Tensor(single) = preserve_elementwise_scalar_shape(
            Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
            &[1, 1, 1],
            "fixture",
        )
        .unwrap() else {
            panic!("Single tensor expected");
        };
        assert_eq!(single.shape(), &[1, 1, 1]);
        assert_eq!(single.numeric_dtype(), runmat_value::NumericDType::F32);
        let Value::ComplexTensor(single) = preserve_elementwise_scalar_shape(
            Value::ComplexTensor(ComplexTensor::from_f32(vec![(0.5, 1.0)], vec![1, 1]).unwrap()),
            &[1, 1, 1],
            "fixture",
        )
        .unwrap() else {
            panic!("complex Single tensor expected");
        };
        assert_eq!(single.shape(), &[1, 1, 1]);
        assert_eq!(single.numeric_dtype(), runmat_value::NumericDType::F32);
        let Value::LogicalArray(logical) = preserve_elementwise_scalar_shape(
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
            &[1, 1, 1],
            "fixture",
        )
        .unwrap() else {
            panic!("logical array expected");
        };
        assert_eq!(logical.shape, vec![1, 1, 1]);
        assert_eq!(logical.data, vec![1]);
        for shape in [vec![0, 1], vec![2, 1], vec![usize::MAX, 2]] {
            let error =
                preserve_elementwise_scalar_shape(Value::Num(1.0), &shape, "fixture").unwrap_err();
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        }
        let output = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let Value::Tensor(unmodified) =
            preserve_elementwise_scalar_shape(output, &[2, 1], "fixture").unwrap()
        else {
            panic!("existing carrier retained");
        };
        assert_eq!(unmodified.shape(), &[1, 2]);
    }
}
