use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

use crate::builtins::common::binary;
use crate::builtins::common::gpu_helpers::{BinaryGpuOutputContract, GpuOutputAliasPolicy};
use crate::BuiltinResult;

use super::super::{errors, SolveOrientation};

pub(super) async fn invoke(
    orientation: SolveOrientation,
    provider: &'static dyn AccelProvider,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    expected_shape: Vec<usize>,
) -> BuiltinResult<Option<Value>> {
    let result = match orientation {
        SolveOrientation::Left => provider.mldivide(lhs, rhs).await,
        SolveOrientation::Right => provider.mrdivide(lhs, rhs).await,
    };
    let output = match result {
        Ok(output) => output,
        Err(error) if error.is_unsupported() => return Ok(None),
        Err(error) => {
            let runtime_error = errors::internal(
                orientation,
                format!("{}: provider solve failed: {error}", orientation.name()),
            );
            return Err(crate::runtime_error::with_provider_execution_source(
                runtime_error,
                error,
            ));
        }
    };
    let contract = BinaryGpuOutputContract {
        shape: expected_shape,
        storage: GpuTensorStorage::Real,
        precision: Some(provider.precision()),
        integer: None,
        logical: false,
        alias: GpuOutputAliasPolicy::RequireDistinct,
    };
    binary::validate_resident_output(provider, lhs, rhs, output, &contract)
        .map(Some)
        .map_err(|source| {
            let mut error = errors::internal(
                orientation,
                format!(
                    "{}: provider returned invalid solve output metadata",
                    orientation.name()
                ),
            );
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error.source = Some(Box::new(source));
            error
        })
}
