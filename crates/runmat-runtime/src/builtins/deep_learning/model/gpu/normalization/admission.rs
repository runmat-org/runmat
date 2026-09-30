//! Admission shared by standalone and planned resident normalization stages.

use super::super::admission::{invalid_owner, GpuOperand};
use super::super::*;

pub(in crate::builtins::deep_learning) struct ResidentParameters<'a> {
    pub scale: Option<&'a GpuTensorHandle>,
    pub offset: Option<&'a GpuTensorHandle>,
}

pub(in crate::builtins::deep_learning) fn admit_parameters<'a>(
    layer: &'a ObjectInstance,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    function: &'static str,
) -> BuiltinResult<Result<ResidentParameters<'a>, GpuCapabilityReason>> {
    let parameters = ResidentParameters {
        scale: resident_parameter_handle(layer, ParameterRole::Scale),
        offset: resident_parameter_handle(layer, ParameterRole::Offset),
    };
    for (role, resident) in [
        (ParameterRole::Scale, parameters.scale),
        (ParameterRole::Offset, parameters.offset),
    ] {
        let Some(resident) = resident else { continue };
        let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(resident)
            .ok_or_else(|| invalid_owner(function, GpuOperand::Parameter(role)))?;
        if !runmat_accelerate_api::same_provider_instance(provider, owner) {
            return Err(invalid_owner(function, GpuOperand::Parameter(role)));
        }
        if runmat_accelerate_api::handle_precision(resident)
            != runmat_accelerate_api::handle_precision(input)
        {
            if runmat_accelerate_api::handle_is_explicit(resident) {
                return Err(unsupported_error(
                    function,
                    format!(
                        "{function}: explicit resident {} precision differs from input",
                        role.name()
                    ),
                ));
            }
            return Ok(Err(GpuCapabilityReason::ParameterPrecision(role)));
        }
    }
    Ok(Ok(parameters))
}
