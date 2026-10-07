use anyhow::Result;
use runmat_accelerate_api::{provider_for_handle, ProviderPrecision};
use runmat_value::{NumericStorage, Tensor, Value};

use super::{WorkspaceMaterializeOptions, WorkspaceSliceOptions};

pub(crate) const MATERIALIZE_DEFAULT_LIMIT: usize = 4096;

pub(crate) fn slice_value_for_preview(
    value: &Value,
    slice: &WorkspaceSliceOptions,
) -> Option<Value> {
    match value {
        Value::Tensor(tensor) => {
            let mut shape = slice.shape.clone();
            if shape.is_empty() {
                shape.push(1);
            }
            let storage = gather_tensor_slice(tensor, slice)?;
            Tensor::from_numeric_storage(storage, shape)
                .ok()
                .map(Value::Tensor)
        }
        _ => None,
    }
}

fn gather_tensor_slice(tensor: &Tensor, slice: &WorkspaceSliceOptions) -> Option<NumericStorage> {
    if tensor.shape().is_empty() || slice.shape.contains(&0) {
        return None;
    }
    let total: usize = slice.shape.iter().product();
    if total == 0 {
        return None;
    }
    let mut indices = Vec::with_capacity(total);
    let mut coords = vec![0usize; tensor.shape().len()];
    visit_slice_coords(tensor.shape(), slice, 0, &mut coords, &mut |coords| {
        indices.push(column_major_index(tensor.shape(), coords));
    });
    if indices.len() != total {
        None
    } else {
        tensor
            .clone()
            .into_numeric_storage()
            .ok()?
            .gather(&indices)
            .ok()
    }
}

fn column_major_index(shape: &[usize], coords: &[usize]) -> usize {
    let mut idx = 0usize;
    let mut stride = 1usize;
    for (dim_len, coord) in shape.iter().zip(coords.iter()) {
        idx += coord * stride;
        stride *= *dim_len;
    }
    idx
}

fn visit_slice_coords<F: FnMut(&[usize])>(
    full_shape: &[usize],
    slice: &WorkspaceSliceOptions,
    axis: usize,
    coords: &mut [usize],
    f: &mut F,
) {
    if axis == full_shape.len() {
        f(coords);
        return;
    }
    let start = slice.start.get(axis).copied().unwrap_or(0);
    let count = slice.shape.get(axis).copied().unwrap_or(1);
    for offset in 0..count {
        coords[axis] = start + offset;
        visit_slice_coords(full_shape, slice, axis + 1, coords, f);
    }
}

pub(crate) fn gpu_dtype_label(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> Option<&'static str> {
    let precision =
        runmat_accelerate_api::handle_precision(handle).unwrap_or(ProviderPrecision::F64);
    match precision {
        ProviderPrecision::F32 => Some("single"),
        ProviderPrecision::F64 => Some("double"),
    }
}

pub(crate) fn gpu_size_bytes(handle: &runmat_accelerate_api::GpuTensorHandle) -> Option<u64> {
    // Report the view's typed payload size, not allocator padding or the
    // provider's preferred arithmetic precision.
    let bytes = runmat_accelerate_api::native_buffer_byte_length(
        handle.shape(),
        handle.physical().element_type(),
        handle.physical().storage(),
    )
    .ok()?;
    u64::try_from(bytes).ok()
}

pub(crate) async fn gather_gpu_preview_values(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    full_shape: &[usize],
    options: &WorkspaceMaterializeOptions,
) -> Result<Option<(Vec<f64>, bool)>> {
    if full_shape.is_empty() || full_shape.contains(&0) {
        return Ok(None);
    }
    let total_elements = full_shape.iter().product::<usize>();
    if total_elements == 0 {
        return Ok(None);
    }

    let provider = provider_for_handle(handle)
        .ok_or_else(|| anyhow::anyhow!("No acceleration provider registered for GPU tensor"))?;

    // Determine which indices to gather.
    let (indices, output_shape, truncated) = if let Some(slice) = options
        .slice
        .as_ref()
        .and_then(|slice| slice.sanitized(full_shape))
    {
        let slice_elements = slice.shape.iter().product::<usize>();
        let requested = slice_elements.min(options.max_elements.max(1));
        let mut indices: Vec<u32> = Vec::with_capacity(requested);
        let mut coords = vec![0usize; full_shape.len()];
        let mut produced = 0usize;
        let mut push_idx = |coords: &[usize]| {
            if produced >= requested {
                return;
            }
            let idx = column_major_index(full_shape, coords);
            if idx <= u32::MAX as usize {
                indices.push(idx as u32);
                produced += 1;
            }
        };
        visit_slice_coords(full_shape, &slice, 0, &mut coords, &mut push_idx);
        let truncated = requested < slice_elements;
        let output_shape = if !truncated && indices.len() == slice_elements {
            slice.shape
        } else {
            vec![indices.len().max(1), 1]
        };
        (indices, output_shape, truncated)
    } else {
        let count = total_elements.min(options.max_elements.max(1));
        let mut indices: Vec<u32> = Vec::with_capacity(count);
        for idx in 0..count {
            if idx > u32::MAX as usize {
                break;
            }
            indices.push(idx as u32);
        }
        let len = indices.len();
        let truncated = total_elements > len;
        (indices, vec![len.max(1), 1], truncated)
    };

    if indices.is_empty() {
        return Ok(None);
    }

    // Gather a small GPU tensor, then download it.
    let gathered = provider
        .gather_linear(handle, &indices, &output_shape)
        .map_err(|e| anyhow::anyhow!("gpu preview gather_linear: {e}"))?;
    let host = provider
        .download(&gathered)
        .await
        .map_err(|e| anyhow::anyhow!("gpu preview download: {e}"))?;
    // Release this preview's reference. Other views retain their own leases;
    // download failure or cancellation releases this reference through Drop.
    drop(gathered);

    Ok(Some((host.data, truncated)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::IntegerStorage;

    #[test]
    fn gpu_preview_size_uses_typed_view_storage() {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, GpuTensorView, NumericElementType,
        };
        let owner = TestAllocationOwner::new();
        for (element, storage, expected) in [
            (NumericElementType::U8, GpuTensorStorage::Real, 6),
            (NumericElementType::I64, GpuTensorStorage::Real, 48),
            (
                NumericElementType::F32,
                GpuTensorStorage::ComplexInterleaved,
                48,
            ),
            (
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
                96,
            ),
        ] {
            let allocation = owner.allocate_zeroed(vec![2, 3], element, storage).unwrap();
            assert_eq!(gpu_size_bytes(&allocation.handle), Some(expected));
            let alias = allocation
                .handle
                .with_view(GpuTensorView::contiguous(vec![3, 2]))
                .unwrap();
            assert_eq!(gpu_size_bytes(&alias), Some(expected));
            drop(allocation.handle);
            assert_eq!(allocation.probe.retirements(), 0);
            drop(alias);
            assert_eq!(allocation.probe.retirements(), 1);
        }
        let empty = owner
            .allocate_zeroed(
                vec![usize::MAX, usize::MAX, 0],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .unwrap();
        assert_eq!(gpu_size_bytes(&empty.handle), Some(0));
    }

    #[test]
    fn preview_slice_preserves_exact_uint64_storage() {
        let values = vec![0, 1, 9_007_199_254_740_993, u64::MAX];
        let tensor = Tensor::new_integer(IntegerStorage::U64(values.clone()), vec![2, 2])
            .expect("typed tensor");
        let slice = WorkspaceSliceOptions {
            start: vec![0, 1],
            shape: vec![2, 1],
        };

        let Value::Tensor(preview) =
            slice_value_for_preview(&Value::Tensor(tensor), &slice).expect("preview")
        else {
            panic!("expected tensor preview");
        };

        assert_eq!(preview.shape(), &[2, 1]);
        assert_eq!(
            preview.integer_storage(),
            Some(&IntegerStorage::U64(vec![values[2], values[3]]))
        );
    }

    #[test]
    fn preview_slice_preserves_native_single_storage() {
        let values = vec![0.1_f32, 0.2, f32::MAX, f32::MIN_POSITIVE];
        let tensor = Tensor::from_f32(values.clone(), vec![2, 2]).expect("single tensor");
        let slice = WorkspaceSliceOptions {
            start: vec![0, 1],
            shape: vec![2, 1],
        };

        let Value::Tensor(preview) =
            slice_value_for_preview(&Value::Tensor(tensor), &slice).expect("preview")
        else {
            panic!("expected tensor preview");
        };
        assert_eq!(preview.shape(), &[2, 1]);
        assert_eq!(
            preview.into_numeric_storage(),
            Ok(NumericStorage::F32(vec![values[2], values[3]]))
        );
    }
}
