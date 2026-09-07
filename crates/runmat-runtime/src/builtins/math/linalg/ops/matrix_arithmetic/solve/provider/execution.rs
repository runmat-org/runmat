use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers::{BinaryGpuOutputContract, GpuOutputAliasPolicy};
use crate::builtins::common::{binary, gpu_helpers};
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
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
        Err(error) => {
            return Err(errors::internal(
                orientation,
                format!("{}: provider solve failed: {error}", orientation.name()),
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
        .map_err(|_| {
            errors::internal(
                orientation,
                format!(
                    "{}: provider returned invalid solve output metadata",
                    orientation.name()
                ),
            )
        })
}
