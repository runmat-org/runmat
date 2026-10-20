use super::super::error::arrayfun_flow;
use super::super::input::ArrayData;
use super::super::output::{maybe_upload_uniform, UniformCollector};
use crate::execution::RootedValue;
use crate::{gather_if_needed_async, BuiltinResult};
use runmat_value::Value;

pub(super) async fn uniform(
    result: RootedValue,
    shape: &[usize],
    inputs: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<RootedValue> {
    if let Value::GpuTensor(handle) = result.value() {
        if handle.shape() == shape {
            return Ok(result);
        }
    }
    // Normalize canonical scalar collapse/host fallback with the ordinary
    // mapping collector, preserving integer widths and complex single storage.
    let data = ArrayData::from_value(gather_if_needed_async(result.value()).await?)?;
    let len = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| arrayfun_flow("arrayfun: output shape exceeds addressable storage"))?;
    let source_len = data
        .shape_vec()
        .iter()
        .try_fold(1usize, |count, d| count.checked_mul(*d));
    if source_len != Some(len) {
        return Err(arrayfun_flow(
            "arrayfun: batched callback returned an unexpected number of elements",
        ));
    }
    let mut collector = UniformCollector::Pending;
    for index in 0..len {
        collector.push(&data.value_at(index)?)?;
    }
    let resident = inputs
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    RootedValue::new(maybe_upload_uniform(collector.finish(shape)?, &resident)?)
        .map_err(|error| error.into_runtime_error())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{ComplexTensor, IntegerStorage, Tensor};

    #[test]
    fn singleton_keeps_full_uniform_array_shape() {
        let result = futures::executor::block_on(uniform(
            RootedValue::new(Value::Num(3.0)).unwrap(),
            &[1, 1, 1],
            crate::call::FunctionArgumentView::from_slice(&[]),
        ))
        .unwrap();
        let Value::Tensor(value) = result.value() else {
            panic!("uniform numeric array")
        };
        assert_eq!(value.shape(), &[1, 1, 1]);
        assert_eq!(
            value.numeric_value_at(0),
            Some(runmat_value::NumericScalar::F64(3.0))
        );
    }

    #[test]
    fn normalization_preserves_wide_integer_and_complex_single() {
        let storage = IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]);
        let input = Value::Tensor(Tensor::new_integer(storage.clone(), vec![1, 2]).unwrap());
        let result = futures::executor::block_on(uniform(
            RootedValue::new(input).unwrap(),
            &[2, 1],
            crate::call::FunctionArgumentView::from_slice(&[]),
        ))
        .unwrap();
        let Value::Tensor(value) = result.value() else {
            panic!("integer array")
        };
        assert_eq!(value.integer_storage(), Some(&storage));
        assert_eq!(value.shape(), &[2, 1]);
        let input =
            Value::ComplexTensor(ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap());
        let result = futures::executor::block_on(uniform(
            RootedValue::new(input).unwrap(),
            &[1, 1, 1],
            crate::call::FunctionArgumentView::from_slice(&[]),
        ))
        .unwrap();
        let Value::ComplexTensor(value) = result.value() else {
            panic!("complex array")
        };
        assert_eq!(value.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(value.shape(), &[1, 1, 1]);
    }
}
