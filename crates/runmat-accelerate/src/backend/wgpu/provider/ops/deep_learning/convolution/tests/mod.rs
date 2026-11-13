use super::*;

fn request() -> DeepLearningConv1DRequest {
    DeepLearningConv1DRequest {
        // SCBT, with deliberately distinct extents on every semantic axis.
        input_shape: vec![7, 3, 2, 4],
        output_shape: vec![4, 5, 2, 4],
        weights_shape: [3, 3, 5],
        bias_shape: [1, 5],
        span_axis: 0,
        channel_axis: 1,
        batch_axis: Some(2),
        time_axis: Some(3),
        stride: 2,
        dilation: 1,
        padding_left: 1,
        padding_right: 1,
        padding_value: DeepLearningConv1DPaddingValue::Replicate,
    }
}

fn upload_native(provider: &WgpuProvider, values: &[f64], shape: &[usize]) -> GpuTensorHandle {
    use runmat_accelerate_api::{AccelProvider, HostNumericDataView, HostNumericTensorView};
    let single;
    let data = match provider.precision {
        NumericPrecision::F32 => {
            single = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
            HostNumericDataView::F32(&single)
        }
        NumericPrecision::F64 => HostNumericDataView::F64(values),
    };
    provider
        .upload_numeric(&HostNumericTensorView {
            data,
            shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

fn download_native(provider: &WgpuProvider, handle: &GpuTensorHandle) -> Vec<f64> {
    use runmat_accelerate_api::{AccelProvider, HostNumericDataOwned};
    match pollster::block_on(provider.download_numeric(handle))
        .unwrap()
        .data
    {
        HostNumericDataOwned::F32(values) => values.into_iter().map(f64::from).collect(),
        HostNumericDataOwned::F64(values) => values,
        other => panic!("expected native real floating storage, got {other:?}"),
    }
}

mod physical;
mod exact;
mod finite_difference;
mod replicate;
mod singleton;
mod reflection;
mod empty;
