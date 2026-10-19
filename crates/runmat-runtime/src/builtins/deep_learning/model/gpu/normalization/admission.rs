//! Admission shared by standalone and planned resident normalization stages.

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
    let scale = match super::super::parameters::admit_resident_parameter(
        layer,
        ParameterRole::Scale,
        provider,
        input,
        function,
    )? {
        Ok(value) => value,
        Err(reason) => return Ok(Err(reason)),
    };
    let offset = match super::super::parameters::admit_resident_parameter(
        layer,
        ParameterRole::Offset,
        provider,
        input,
        function,
    )? {
        Ok(value) => value,
        Err(reason) => return Ok(Err(reason)),
    };
    Ok(Ok(ResidentParameters { scale, offset }))
}
