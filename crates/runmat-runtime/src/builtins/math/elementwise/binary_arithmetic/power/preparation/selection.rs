//! Original input roots and passive branch data; no sibling future is built.
use super::*;
use crate::builtins::math::host_route::{self, NumericHostRoute};
pub(super) struct Pair {
    pub(super) left: Value,
    pub(super) right: Value,
}
pub(super) enum Branch {
    Host(NumericHostRoute),
    Admitted(BuiltinPreparationAdmission),
    Ordinary,
}
pub(super) struct Selected {
    pub(super) pair: Pair,
    pub(super) branch: Branch,
    pub(super) inputs: RootedValueLease,
}
pub(super) fn select(
    runtime: Option<&RuntimeContext>,
    input_count: usize,
    left: Value,
    right: Value,
) -> BuiltinResult<Selected> {
    crate::builtins::common::gpu_helpers::require_resident_compute_values(
        &[&left, &right],
        false,
        || super::super::builtin_error("power: resident inputs have no exact owning provider"),
    )?;
    let inputs = RootedValueLease::for_values([&left, &right].into_iter())
        .map_err(|cause| cause.into_runtime_error())?;
    if input_count == 2 {
        if let Some(route) = host_route::canonical(
            runtime,
            crate::context::VmFusionHostInstruction::ElementwiseBinary,
            [&left, &right].into_iter(),
        ) {
            return Ok(Selected {
                pair: Pair { left, right },
                branch: Branch::Host(route),
                inputs,
            });
        }
    }
    let resident = [&left, &right]
        .into_iter()
        .any(|v| matches!(v, Value::GpuTensor(_)));
    let admitted = input_count == 2
        && [&left, &right].into_iter().all(|value| match value {
            Value::GpuTensor(_) => host_route::real_floating_shape(value).is_some(),
            Value::Tensor(t) => matches!(t.numeric_dtype(), NumericDType::F32 | NumericDType::F64),
            Value::Num(_) => resident,
            _ => false,
        });
    let branch = if admitted {
        Branch::Admitted(BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: POWER_CATALOG_ENTRY.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        })
    } else {
        Branch::Ordinary
    };
    Ok(Selected {
        pair: Pair { left, right },
        branch,
        inputs,
    })
}
pub(super) struct Request {
    pub(super) request: AdmittedBuiltinPreparation,
    pub(super) direct: Option<AdmittedBuiltinCompound>,
}
pub(super) fn request(
    runtime: Option<&RuntimeContext>,
    admission: BuiltinPreparationAdmission,
    pair: Pair,
) -> BuiltinResult<Request> {
    let Pair { left, right } = pair;
    let request = AdmittedBuiltinPreparation::new(admission, vec![left, right])?
        .with_real_power()?
        .with_numerical_domain(classify)?;
    let direct = if runtime.is_none_or(|r| r.service_ports().acceleration().is_none()) {
        request
            .operands()
            .iter()
            .find_map(|value| match value {
                Value::GpuTensor(source) => {
                    crate::builtins::common::gpu_helpers::exact_provider_for_handle(source)
                }
                _ => None,
            })
            .map(|owner| request.plan_compound(runtime, owner))
            .transpose()?
            .flatten()
    } else {
        None
    };
    Ok(Request { request, direct })
}
