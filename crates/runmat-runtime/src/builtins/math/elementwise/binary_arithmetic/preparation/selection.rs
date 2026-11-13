//! Inert selected data, not an eagerly constructed sibling body future.
use super::*;
use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::math::host_route::{self, NumericHostRoute};
use crate::context::preparation::BuiltinPreparationAdmission;
use crate::execution::RootedValueLease;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId};

pub(super) enum Branch {
    Host(NumericHostRoute),
    Admitted(BuiltinPreparationAdmission),
    Ordinary,
}
pub(super) struct Selected {
    pub(super) left: Value,
    pub(super) right: Value,
    pub(super) branch: Branch,
    // Drop the input headers/routes before retiring the original root forest.
    pub(super) inputs: RootedValueLease,
}

fn real_arithmetic_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Bool(_) => Some(&[1, 1]),
        Value::LogicalArray(logical) => Some(logical.shape.as_slice()),
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_is_logical(handle)
                && handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real
                && matches!(
                    handle.physical().element_type(),
                    runmat_accelerate_api::NumericElementType::U8
                        | runmat_accelerate_api::NumericElementType::F32
                        | runmat_accelerate_api::NumericElementType::F64
                )
                && runmat_accelerate_api::handle_provenance(handle)
                    == Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                && handle.allocation_lease().validate_live().is_ok()
                && handle.view().validate(handle.physical()).is_ok()
                && crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
                    .is_some() =>
        {
            Some(handle.shape())
        }
        other => host_route::real_floating_shape(other),
    }
}

pub(super) fn prepare(
    context: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    operator: runmat_types::OperatorKind,
    input_count: usize,
    left: Value,
    right: Value,
) -> BuiltinResult<Selected> {
    if let Some((runtime, route)) = context.and_then(|runtime| {
        runtime
            .native_region_host_route()
            .map(|route| (runtime, route))
    }) {
        if input_count != 2
            || !host_route::real_floating_host(&left)
            || !host_route::real_floating_host(&right)
        {
            return Err(managed::invalid());
        }
        route.require_binary_host(
            runtime,
            operator,
            BuiltinBindingIdentity {
                builtin: identity,
                variant: "default",
            },
        )?;
    }
    let inputs = RootedValueLease::for_values([&left, &right].into_iter())
        .map_err(|cause| cause.into_runtime_error())?;
    if input_count == 2 {
        if let Some(route) = host_route::canonical(
            context,
            crate::context::VmFusionHostInstruction::ElementwiseBinary,
            [&left, &right].into_iter(),
        ) {
            return Ok(Selected {
                left,
                right,
                branch: Branch::Host(route),
                inputs,
            });
        }
    }
    let admitted = input_count == 2
        && real_arithmetic_shape(&left)
            .zip(real_arithmetic_shape(&right))
            .is_some_and(|(left, right)| BroadcastPlan::new(left, right).is_ok());
    let branch = if admitted {
        Branch::Admitted(BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        })
    } else {
        // Original body owns other representations/errors. Invalid shapes do
        // not enter the service or upload either operand.
        Branch::Ordinary
    };
    Ok(Selected {
        left,
        right,
        branch,
        inputs,
    })
}
