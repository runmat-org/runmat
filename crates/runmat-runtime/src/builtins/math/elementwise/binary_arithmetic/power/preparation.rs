//! Value-domain evidence belongs to this admitted pair, not a cached name.
use crate::context::preparation::{
    execute_request_numerical, AdmittedBuiltinCompound, AdmittedBuiltinPreparation,
    BuiltinPreparationAdmission,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId, POWER_CATALOG_ENTRY};
use runmat_value::{NumericDType, Value};
use std::future::Future;

pub(super) async fn execute<F, Fut, P, Published>(
    runtime: Option<&RuntimeContext>,
    input_count: usize,
    left: Value,
    right: Value,
    body: F,
    publish: P,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value, Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
    P: FnOnce(RootedValue) -> Published,
    Published: Future<Output = BuiltinResult<RootedValue>>,
{
    let _inputs = RootedValueLease::for_values([&left, &right].into_iter())
        .map_err(|cause| cause.into_runtime_error())?;
    if input_count == 2 {
        if let Some(route) = crate::builtins::math::host_route::canonical(
            runtime,
            crate::context::VmFusionHostInstruction::ElementwiseBinary,
            [&left, &right].into_iter(),
        ) {
            let numerical_output = route.take_numerical_output(
                runtime.unwrap(),
                crate::context::VmFusionHostInstruction::ElementwiseBinary,
                BuiltinBindingIdentity {
                    builtin: POWER_CATALOG_ENTRY.identity,
                    variant: "default",
                },
            )?;
            let (output, evidence) = match numerical_output {
                Some(output) => {
                    let (output, domain) =
                        managed::execute(runtime.unwrap(), output, &left, &right)?;
                    (
                        output,
                        runmat_accelerate_api::NumericalDomainEvidence::RealPower(domain),
                    )
                }
                None => {
                    let evidence = classify_operands(&left, &right)?;
                    (body(left, right).await?, evidence)
                }
            };
            let output = publish(output).await?;
            route.confirm_numerical_host_branch(
                runtime.unwrap(),
                crate::context::VmFusionHostInstruction::ElementwiseBinary,
                evidence,
            );
            return Ok(output);
        }
    }
    let resident = [&left, &right]
        .into_iter()
        .any(|value| matches!(value, Value::GpuTensor(_)));
    let admitted = input_count == 2
        && [&left, &right].into_iter().all(|value| match value {
            Value::GpuTensor(_) => {
                crate::builtins::math::host_route::real_floating_shape(value).is_some()
            }
            Value::Tensor(t) => matches!(t.numeric_dtype(), NumericDType::F32 | NumericDType::F64),
            Value::Num(_) => resident,
            _ => false,
        });
    if !admitted {
        return publish(body(left, right).await?).await;
    }
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: POWER_CATALOG_ENTRY.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![left, right],
    )?
    .with_real_power()?
    .with_numerical_domain(classify)?;
    let direct = if runtime.is_none_or(|runtime| runtime.service_ports().acceleration().is_none()) {
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
    execute_request_numerical(
        runtime,
        request,
        |mut values, compound, numerical_output| async move {
            if let Some(output) = numerical_output {
                if compound.is_some() || direct.is_some() {
                    return Err(super::super::preparation::managed::invalid());
                }
                let right = values.remove(1);
                let left = values.remove(0);
                let route = crate::builtins::math::host_route::canonical(
                    runtime,
                    crate::context::VmFusionHostInstruction::ElementwiseBinary,
                    [&left, &right].into_iter(),
                );
                let (output, domain) = managed::execute(
                    runtime.ok_or_else(super::super::preparation::managed::invalid)?,
                    output,
                    &left,
                    &right,
                )?;
                let output = publish(output).await?;
                if let Some(route) = route {
                    route.confirm_numerical_host_branch(
                        runtime.unwrap(),
                        crate::context::VmFusionHostInstruction::ElementwiseBinary,
                        runmat_accelerate_api::NumericalDomainEvidence::RealPower(domain),
                    );
                }
                return Ok(output);
            }
            if let Some(compound) = compound.or(direct) {
                let right = values.remove(1);
                let left = values.remove(0);
                let output = match compound {
                    AdmittedBuiltinCompound::RealPower(compound) => {
                        super::compound::execute(runtime, *compound, left, right).await?
                    }
                    AdmittedBuiltinCompound::RealPowerScalar(compound) => {
                        super::scalar_compound::execute(runtime, *compound, left, right).await?
                    }
                    AdmittedBuiltinCompound::RealUnary(_) => {
                        let mut error = super::builtin_error(
                            "power: received another admitted compound authority",
                        );
                        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                        return Err(error);
                    }
                };
                return publish(output).await;
            }
            let route = crate::builtins::math::host_route::canonical(
                runtime,
                crate::context::VmFusionHostInstruction::ElementwiseBinary,
                values.iter(),
            );
            if let (Some(_), Some(runtime)) = (&route, runtime) {
                super::super::preparation::managed::require_unbounded_host_output(runtime)?;
            }
            if route.is_none() {
                // Missing complete physical recipe is not a certified no-attempt
                // hook result. Validate each actual owner before legacy probe or
                // scalar-staging effects; never invent an owner for unknown views.
                let mut owners: Vec<&'static dyn runmat_accelerate_api::AccelProvider> = Vec::new();
                for value in &values {
                    let Value::GpuTensor(source) = value else {
                        continue;
                    };
                    let Some(owner) =
                        crate::builtins::common::gpu_helpers::exact_provider_for_handle(source)
                    else {
                        continue;
                    };
                    if owners.iter().any(|previous| {
                        runmat_accelerate_api::same_provider_instance(*previous, owner)
                    }) {
                        continue;
                    }
                    owners.push(owner);
                    crate::context::provider_operation::unavailable_provider_route::<()>(
                        runtime, owner,
                    )?;
                }
            }
            let evidence = route.as_ref().map(|_| classify(&values)).transpose()?;
            let right = values.remove(1);
            let output = publish(body(values.remove(0), right).await?).await?;
            if let (Some(route), Some(evidence)) = (route, evidence) {
                route.confirm_numerical_host_branch(
                    runtime.unwrap(),
                    crate::context::VmFusionHostInstruction::ElementwiseBinary,
                    evidence,
                );
            }
            Ok(output)
        },
    )
    .await
}

#[path = "preparation/domain.rs"]
mod domain;
use domain::{classify, classify_operands};
mod managed;
