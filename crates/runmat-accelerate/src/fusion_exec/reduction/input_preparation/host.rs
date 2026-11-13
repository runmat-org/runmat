//! Scoped host word conversion and exact typed upload.
use super::invalid;
use crate::fusion_exec::outputs::FusionOutputAdmission;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
    ProviderExecutionResult, ProviderPrecision,
};
use runmat_value::Value;

pub(super) fn upload(
    admission: &mut FusionOutputAdmission,
    provider: &'static dyn AccelProvider,
    source: &Value,
    target: ProviderPrecision,
) -> ProviderExecutionResult<GpuTensorHandle> {
    let shape = match source {
        Value::Num(_) => &[1, 1][..],
        Value::Tensor(tensor) => tensor.shape(),
        _ => return Err(invalid("typed upload source changed")),
    };
    match (source, target) {
        (Value::Num(value), runmat_accelerate_api::ProviderPrecision::F64) => admission
            .upload_typed(
                provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::F64(std::slice::from_ref(value)),
                    shape,
                    storage: GpuTensorStorage::Real,
                },
            ),
        (Value::Num(value), runmat_accelerate_api::ProviderPrecision::F32) => {
            let words = [*value as f32];
            admission.upload_typed(
                provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::F32(&words),
                    shape,
                    storage: GpuTensorStorage::Real,
                },
            )
        }
        (Value::Tensor(tensor), runmat_accelerate_api::ProviderPrecision::F32) => {
            let owned;
            let words = if let Some(exact) = tensor.as_f32_slice() {
                exact
            } else {
                owned = tensor
                    .as_f64_slice()
                    .ok_or_else(|| invalid("typed host source class changed"))?
                    .iter()
                    .map(|&value| value as f32)
                    .collect::<Vec<_>>();
                &owned
            };
            admission.upload_typed(
                provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::F32(words),
                    shape,
                    storage: GpuTensorStorage::Real,
                },
            )
        }
        (Value::Tensor(tensor), runmat_accelerate_api::ProviderPrecision::F64) => {
            let owned;
            let words = if let Some(exact) = tensor.as_f64_slice() {
                exact
            } else {
                owned = tensor
                    .as_f32_slice()
                    .ok_or_else(|| invalid("typed host source class changed"))?
                    .iter()
                    .map(|&value| f64::from(value))
                    .collect::<Vec<_>>();
                &owned
            };
            admission.upload_typed(
                provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::F64(words),
                    shape,
                    storage: GpuTensorStorage::Real,
                },
            )
        }
        _ => Err(invalid("typed upload target or source unavailable")),
    }
}
