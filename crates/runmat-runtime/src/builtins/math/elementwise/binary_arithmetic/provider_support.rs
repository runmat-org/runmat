use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::BuiltinCatalogIdentity;
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::BuiltinResult;

pub(super) fn broadcast_repetitions(
    left: &[usize],
    right: &[usize],
) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let rank = left.len().max(right.len()).max(1);
    let left = crate::builtins::common::broadcast::align_shape(left, rank);
    let right = crate::builtins::common::broadcast::align_shape(right, rank);
    let mut output = Vec::with_capacity(rank);
    for (&left_extent, &right_extent) in left.iter().zip(&right) {
        output.push(match (left_extent, right_extent) {
            (left, right) if left == right => left,
            (1, right) => right,
            (left, 1) => left,
            _ => return None,
        });
    }
    let left_repetitions = repetitions(&left, &output);
    let right_repetitions = repetitions(&right, &output);
    Some((output, left_repetitions, right_repetitions))
}

pub(super) fn host_real_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(value)
            if tensor::is_scalar_tensor(value) && value.integer_storage().is_none() =>
        {
            Some(tensor::tensor_value_f64(value, 0))
        }
        Value::LogicalArray(value) if value.data.len() == 1 => {
            Some(if value.data[0] != 0 { 1.0 } else { 0.0 })
        }
        Value::CharArray(value) if value.rows * value.cols == 1 => Some(
            value
                .data
                .first()
                .map(|&character| character as u32 as f64)
                .unwrap_or(0.0),
        ),
        _ => None,
    }
}

pub(super) async fn device_real_scalar(
    identity: BuiltinCatalogIdentity,
    handle: &GpuTensorHandle,
) -> BuiltinResult<Option<f64>> {
    if !is_scalar_shape(&handle.shape)
        || runmat_accelerate_api::handle_integer_type(handle).is_some()
    {
        return Ok(None);
    }
    let tensor = gpu_helpers::gather_tensor_async(handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, identity.name))?;
    Ok(tensor::tensor_values_f64(&tensor).first().copied())
}

pub(super) fn is_scalar_shape(shape: &[usize]) -> bool {
    shape.iter().copied().product::<usize>() <= 1
}

pub(super) fn resident_output_from_sources<'a>(
    mut output: GpuTensorHandle,
    sources: impl IntoIterator<Item = &'a GpuTensorHandle>,
) -> Value {
    let provenance = if sources
        .into_iter()
        .any(runmat_accelerate_api::handle_is_explicit)
    {
        runmat_accelerate_api::GpuHandleProvenance::Explicit
    } else {
        runmat_accelerate_api::GpuHandleProvenance::Automatic
    };
    runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
    gpu_helpers::resident_gpu_value(output)
}

fn repetitions(shape: &[usize], output: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .zip(output)
        .map(|(&extent, &output_extent)| {
            if extent == output_extent {
                1
            } else {
                output_extent
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::GpuHandleProvenance;
    use runmat_value::{NumericStorage, Tensor};

    #[test]
    fn broadcast_repetitions_aligns_ranks_and_singletons() {
        let (shape, left, right) =
            broadcast_repetitions(&[3, 1], &[1, 4]).expect("compatible shapes");
        assert_eq!(shape, vec![3, 4]);
        assert_eq!(left, vec![1, 4]);
        assert_eq!(right, vec![3, 1]);
        assert!(broadcast_repetitions(&[2, 3], &[4, 3]).is_none());
    }

    #[test]
    fn provider_scalar_fast_path_does_not_approximate_integer_storage() {
        let integer = Tensor::from_numeric_storage(NumericStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("integer scalar");
        assert_eq!(host_real_scalar(&Value::Tensor(integer)), None);

        let floating = Tensor::new(vec![3.5], vec![1, 1]).expect("floating scalar");
        assert_eq!(host_real_scalar(&Value::Tensor(floating)), Some(3.5));
    }

    #[test]
    fn empty_and_single_element_shapes_are_scalar_provider_shapes() {
        assert!(is_scalar_shape(&[]));
        assert!(is_scalar_shape(&[1, 1, 1]));
        assert!(!is_scalar_shape(&[1, 2]));
    }

    #[test]
    fn resident_outputs_preserve_explicit_source_intent() {
        let automatic =
            GpuTensorHandle::new(vec![2, 2], 1, 1).with_provenance(GpuHandleProvenance::Automatic);
        let explicit =
            GpuTensorHandle::new(vec![2, 2], 1, 2).with_provenance(GpuHandleProvenance::Explicit);
        let output = GpuTensorHandle::new(vec![2, 2], 1, 3);

        let Value::GpuTensor(output) =
            resident_output_from_sources(output, [&automatic, &explicit])
        else {
            panic!("resident output must remain a GPU tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&output),
            Some(GpuHandleProvenance::Explicit)
        );
    }
}
