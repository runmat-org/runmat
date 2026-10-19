//! Native-class, owner-protected publication of host-computed eig factors.

use super::*;
use runmat_accelerate_api::{
    upload_numeric_protecting, HostNumericDataView, HostNumericTensorView, NumericUploadError,
};
use runmat_value::{ComplexTensor, NumericDType};

fn floating_precision(
    dtype: NumericDType,
) -> runmat_accelerate_api::ProviderExecutionResult<ProviderPrecision> {
    match dtype {
        NumericDType::F32 => Ok(ProviderPrecision::F32),
        NumericDType::F64 => Ok(ProviderPrecision::F64),
        _ => Err(runmat_accelerate_api::unsupported_provider_operation(
            "eig output must have single or double floating class",
        )
        .into()),
    }
}

pub(super) fn admit_eig_output_class(
    value: &Value,
    source_precision: ProviderPrecision,
) -> runmat_accelerate_api::ProviderExecutionResult<()> {
    let dtype = match value {
        Value::Tensor(tensor) => tensor.numeric_dtype(),
        Value::ComplexTensor(tensor) => tensor.numeric_dtype(),
        Value::Num(_) | Value::Complex(_, _) => NumericDType::F64,
        _ => {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "eig produced a nonnumeric factor",
            )
            .into())
        }
    };
    if floating_precision(dtype)? != source_precision {
        return Err(anyhow!("eig output class differs from the admitted source class").into());
    }
    Ok(())
}

impl WgpuProvider {
    pub(super) fn upload_eig_output_exec(
        &self,
        value: Value,
        protected: &[&GpuTensorHandle],
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        match value {
            Value::Tensor(tensor) => {
                let data = if let Some(values) = tensor.as_f32_slice() {
                    HostNumericDataView::F32(values)
                } else if let Some(values) = tensor.as_f64_slice() {
                    HostNumericDataView::F64(values)
                } else {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "eig factor must have floating storage",
                    )
                    .into());
                };
                upload_numeric_protecting(
                    self,
                    &HostNumericTensorView {
                        data,
                        shape: tensor.shape(),
                        storage: GpuTensorStorage::Real,
                    },
                    protected,
                )
                .map_err(NumericUploadError::into_execution_error)
            }
            Value::ComplexTensor(tensor) => self.upload_complex_eig_output_exec(&tensor, protected),
            Value::Num(number) => {
                let tensor = Tensor::new(vec![number], vec![1, 1])
                    .map_err(|error| anyhow!("eig scalar factor: {error}"))?;
                self.upload_eig_output_exec(Value::Tensor(tensor), protected)
            }
            Value::Complex(real, imaginary) => {
                let tensor = ComplexTensor::new(vec![(real, imaginary)], vec![1, 1])
                    .map_err(|error| anyhow!("eig complex scalar factor: {error}"))?;
                self.upload_complex_eig_output_exec(&tensor, protected)
            }
            _ => Err(runmat_accelerate_api::unsupported_provider_operation(
                "eig produced a nonnumeric factor",
            )
            .into()),
        }
    }

    fn upload_complex_eig_output_exec(
        &self,
        tensor: &ComplexTensor,
        protected: &[&GpuTensorHandle],
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if let Some(values) = tensor.as_f32_slice() {
            let interleaved = values
                .iter()
                .flat_map(|value| [value.0, value.1])
                .collect::<Vec<_>>();
            upload_numeric_protecting(
                self,
                &HostNumericTensorView {
                    data: HostNumericDataView::F32(&interleaved),
                    shape: tensor.shape(),
                    storage: GpuTensorStorage::ComplexInterleaved,
                },
                protected,
            )
            .map_err(NumericUploadError::into_execution_error)
        } else if let Some(values) = tensor.as_f64_slice() {
            let interleaved = values
                .iter()
                .flat_map(|value| [value.0, value.1])
                .collect::<Vec<_>>();
            upload_numeric_protecting(
                self,
                &HostNumericTensorView {
                    data: HostNumericDataView::F64(&interleaved),
                    shape: tensor.shape(),
                    storage: GpuTensorStorage::ComplexInterleaved,
                },
                protected,
            )
            .map_err(NumericUploadError::into_execution_error)
        } else {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "eig complex factor must have floating storage",
            )
            .into())
        }
    }
}
