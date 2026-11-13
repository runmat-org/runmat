//! Preparation of an admitted real floating operand with request-local domain evidence.
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::preparation::{
    execute_request_numerical, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_accelerate_api::NumericalDomainEvidence;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinPreparationFormId};
use runmat_types::{RealUnaryClassification, RealUnaryPredicate};
use runmat_value::{NumericDType, Tensor, Value};
use std::future::Future;
mod continuation;
pub use continuation::RealUnaryPublication;
mod runner;
pub(crate) use runner::{BeforeHost, DevicePublication, HostPublication, RealUnaryAdapter};

pub(crate) async fn execute<F, Fut, B, H, R, F32, F64>(
    runtime: Option<&RuntimeContext>,
    entry: &'static BuiltinCatalogEntry,
    operation: runmat_types::ScalarUnaryMath,
    publication: RealUnaryPublication,
    adapter: RealUnaryAdapter<B, H, R>,
    value: Value,
    complex_on_outside: bool,
    mut host_f32: F32,
    mut host_f64: F64,
    body: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
    B: BeforeHost,
    H: HostPublication,
    R: DevicePublication,
    F32: FnMut(f32) -> BuiltinResult<(f32, f32)>,
    F64: FnMut(f64) -> BuiltinResult<(f64, f64)>,
{
    let predicate = operation.real_predicate().ok_or_else(|| {
        crate::build_runtime_error("real unary preparation requires a promoting-domain instruction")
            .with_identifier("RunMat:Placement:InvalidRequest")
            .build()
    })?;
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    let managed = match runtime {
        Some(context) => {
            HostAllocationReceipt::policy(Some(context)).map_err(|error| {
                crate::builtin::builtin_control_error(
                    Some(&context.cancellation_context()),
                    entry.identity.name,
                    error,
                )
            })? == HostAllocationPolicy::Managed
        }
        None => false,
    };
    if let Value::GpuTensor(source) = &value {
        crate::builtins::common::gpu_helpers::require_resident_compute_owner(source, || {
            crate::build_runtime_error("real unary input has no exact compute owner")
                .with_builtin(entry.identity.name)
                .build()
        })?;
    }
    if let Some(route) = crate::builtins::math::host_route::canonical(
        runtime,
        crate::context::VmFusionHostInstruction::ElementwiseCall,
        std::iter::once(&value),
    ) {
        let context = runtime.ok_or_else(invalid_selected)?;
        if managed {
            let output = route
                .take_numerical_output(
                    context,
                    crate::context::VmFusionHostInstruction::ElementwiseCall,
                    BuiltinBindingIdentity {
                        builtin: entry.identity,
                        variant: "default",
                    },
                )?
                .ok_or_else(|| {
                    crate::builtin::builtin_control_error(
                        Some(&context.cancellation_context()),
                        entry.identity.name,
                        HostAllocationError::MissingManagedOwner,
                    )
                })?;
            let evidence = classify_value(&value, predicate, runtime, entry.identity.name)?;
            let outside = matches!(
                evidence,
                NumericalDomainEvidence::RealUnary {
                    classification: RealUnaryClassification::Outside,
                    ..
                }
            );
            let result = super::host_unary_output::promoting(
                context,
                output,
                runmat_builtins::BuiltinPreparationOperation::RealUnary(predicate),
                &value,
                outside && complex_on_outside,
                &mut host_f32,
                &mut host_f64,
            )?;
            let result = RootedValue::new(result).map_err(|cause| cause.into_runtime_error())?;
            route.confirm_numerical_host_branch(
                context,
                crate::context::VmFusionHostInstruction::ElementwiseCall,
                evidence,
            );
            return Ok(result);
        } else {
            // The admitted host operand reaches the existing host-only numeric
            // body. Domain diagnostics and complex-result semantics stay there;
            // only provider-oriented preparation/promotion is bypassed.
            let evidence = classify_value(&value, predicate, runtime, entry.identity.name)?;
            let output = body(value).await?;
            route.confirm_numerical_host_branch(
                context,
                crate::context::VmFusionHostInstruction::ElementwiseCall,
                evidence,
            );
            return Ok(output);
        }
    }
    let resident = matches!(&value, Value::GpuTensor(_)
        if crate::builtins::math::host_route::real_floating_shape(&value).is_some());
    if !managed
        && !resident
        && !matches!(&value, Value::Tensor(tensor) if matches!(tensor.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
    {
        if let Value::GpuTensor(source) = &value {
            if let Some(owner) =
                crate::builtins::common::gpu_helpers::exact_provider_for_handle(source)
            {
                crate::context::provider_operation::unavailable_provider_route::<()>(
                    runtime, owner,
                )?;
            }
        }
        return body(value).await;
    }
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 1,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
    )?
    .with_real_unary(operation, publication)?
    .with_numerical_domain(|operands| match operands {
        [Value::Num(number)] => {
            check_classification(runtime, entry.identity.name)?;
            Ok(NumericalDomainEvidence::RealUnary {
                predicate,
                classification: classify_values(std::iter::once(*number), predicate),
            })
        }
        [Value::Tensor(tensor)] => Ok(NumericalDomainEvidence::RealUnary {
            predicate,
            classification: classify_tensor(tensor, predicate, runtime, entry.identity.name)?,
        }),
        [Value::GpuTensor(_)] if resident => Ok(NumericalDomainEvidence::Unknown),
        _ => unreachable!("admitted floating operand"),
    })?;
    if !managed
        && !resident
        && !matches!(
            request.numerical_domain(),
            NumericalDomainEvidence::RealUnary {
                classification: RealUnaryClassification::Inside,
                ..
            }
        )
    {
        return request
            .into_operands()
            .with_unbounded_values_async(|mut values| body(values.remove(0)))
            .await?;
    }
    let direct = if runtime.is_none_or(|runtime| runtime.service_ports().acceleration().is_none()) {
        match request.operands() {
            [Value::GpuTensor(source)] => {
                match crate::builtins::common::gpu_helpers::exact_provider_for_handle(source) {
                    Some(owner) => request.plan_compound(runtime, owner)?,
                    None => None,
                }
            }
            _ => None,
        }
    } else {
        None
    };
    execute_request_numerical(
        runtime,
        request,
        |mut values, compound, numerical_output| async move {
            let value = values.remove(0);
            if compound.is_some() && numerical_output.is_some() {
                return Err(invalid_selected());
            }
            if let Some(compound) = compound.or(direct) {
                let crate::context::preparation::AdmittedBuiltinCompound::RealUnary(compound) =
                    compound
                else {
                    return Err(crate::build_runtime_error(
                        "unary body received another compound authority",
                    )
                    .with_identifier("RunMat:Placement:InvalidDecision")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build());
                };
                if compound.admission().binding.builtin != entry.identity
                    || compound.unary().operation() != operation
                    || compound.unary().publication() != publication
                {
                    return Err(crate::build_runtime_error(
                        "compound unary delivery differs from its admitted body",
                    )
                    .with_identifier("RunMat:Placement:InvalidDecision")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build());
                }
                return runner::execute(runtime, *compound, value, body, adapter).await;
            }
            if let Some(output) = numerical_output {
                let Some(context) = runtime else {
                    return Err(invalid_selected());
                };
                let evidence = classify_value(&value, predicate, runtime, entry.identity.name)?;
                let outside = matches!(
                    evidence,
                    NumericalDomainEvidence::RealUnary {
                        classification: RealUnaryClassification::Outside,
                        ..
                    }
                );
                let route = crate::builtins::math::host_route::canonical(
                    runtime,
                    crate::context::VmFusionHostInstruction::ElementwiseCall,
                    std::iter::once(&value),
                );
                let output = super::host_unary_output::promoting(
                    context,
                    output,
                    runmat_builtins::BuiltinPreparationOperation::RealUnary(predicate),
                    &value,
                    outside && complex_on_outside,
                    &mut host_f32,
                    &mut host_f64,
                )?;
                let output =
                    RootedValue::new(output).map_err(|cause| cause.into_runtime_error())?;
                if let Some(route) = route {
                    route.confirm_numerical_host_branch(
                        context,
                        crate::context::VmFusionHostInstruction::ElementwiseCall,
                        evidence,
                    );
                }
                return Ok(output);
            }
            if let Value::GpuTensor(source) = &value {
                if let Some(owner) =
                    crate::builtins::common::gpu_helpers::exact_provider_for_handle(source)
                {
                    // No complete physical body was delivered. None is not a
                    // permission to bypass a finite budget with legacy probe hooks.
                    crate::context::provider_operation::unavailable_provider_route::<()>(
                        runtime, owner,
                    )?;
                }
            }
            let route = crate::builtins::math::host_route::canonical(
                runtime,
                crate::context::VmFusionHostInstruction::ElementwiseCall,
                std::iter::once(&value),
            );
            if managed && route.is_some() && matches!(&value, Value::Num(_) | Value::Tensor(_)) {
                let context = runtime.ok_or_else(invalid_selected)?;
                return Err(crate::builtin::builtin_control_error(
                    Some(&context.cancellation_context()),
                    entry.identity.name,
                    HostAllocationError::MissingManagedOwner,
                ));
            }
            let evidence = if route.is_some() {
                Some(classify_value(
                    &value,
                    predicate,
                    runtime,
                    entry.identity.name,
                )?)
            } else {
                None
            };
            let output = body(value).await?;
            if let (Some(route), Some(evidence)) = (route, evidence) {
                route.confirm_numerical_host_branch(
                    runtime.unwrap(),
                    crate::context::VmFusionHostInstruction::ElementwiseCall,
                    evidence,
                );
            }
            Ok(output)
        },
    )
    .await
}

fn invalid_selected() -> crate::RuntimeError {
    crate::build_runtime_error("selected real unary output lacks its admitted context")
        .with_identifier("RunMat:Placement:InvalidDecision")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn classify_tensor(
    tensor: &Tensor,
    predicate: RealUnaryPredicate,
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
) -> BuiltinResult<RealUnaryClassification> {
    const CANCELLATION_CHUNK: usize = 4_096;

    check_classification(runtime, builtin)?;
    let mut classification = RealUnaryClassification::Inside;
    for start in (0..tensor.len()).step_by(CANCELLATION_CHUNK) {
        check_classification(runtime, builtin)?;
        let end = tensor.len().min(start.saturating_add(CANCELLATION_CHUNK));
        let chunk = classify_values(
            (start..end)
                .map(|index| crate::builtins::common::tensor::tensor_value_f64(tensor, index)),
            predicate,
        );
        check_classification(runtime, builtin)?;
        match chunk {
            RealUnaryClassification::Outside => {
                return Ok(RealUnaryClassification::Outside);
            }
            RealUnaryClassification::Indeterminate => {
                classification = RealUnaryClassification::Indeterminate;
            }
            RealUnaryClassification::Inside => {}
        }
    }
    Ok(classification)
}

fn classify_value(
    value: &Value,
    predicate: RealUnaryPredicate,
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
) -> BuiltinResult<NumericalDomainEvidence> {
    let classification = match value {
        Value::Tensor(tensor) => classify_tensor(tensor, predicate, runtime, builtin)?,
        Value::Num(value) => {
            check_classification(runtime, builtin)?;
            classify_values(std::iter::once(*value), predicate)
        }
        _ => unreachable!("canonical native floating host branch"),
    };
    Ok(NumericalDomainEvidence::RealUnary {
        predicate,
        classification,
    })
}

fn check_classification(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let Some(runtime) = runtime else {
        return Ok(());
    };
    let cancellation = runtime.cancellation_context();
    if cancellation.is_cancelled() {
        return Err(crate::builtin::builtin_control_error(
            Some(&cancellation),
            builtin,
            HostAllocationError::Cancelled,
        ));
    }
    Ok(())
}

fn classify_values(
    values: impl Iterator<Item = f64>,
    predicate: RealUnaryPredicate,
) -> RealUnaryClassification {
    predicate.classify(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundaries_nan_and_single_values_are_classified_without_rounding() {
        for (values, predicate, expected) in [
            (
                vec![-0.0, f64::INFINITY],
                RealUnaryPredicate::Nonnegative,
                RealUnaryClassification::Inside,
            ),
            (
                vec![-1.0, 0.0],
                RealUnaryPredicate::AtLeastMinusOne,
                RealUnaryClassification::Inside,
            ),
            (
                vec![0.0, f64::NAN],
                RealUnaryPredicate::Nonnegative,
                RealUnaryClassification::Indeterminate,
            ),
            (
                vec![f64::NAN, -1e-300],
                RealUnaryPredicate::Nonnegative,
                RealUnaryClassification::Outside,
            ),
        ] {
            let tensor = Tensor::new(values, vec![1, 2]).unwrap();
            assert_eq!(
                classify_tensor(&tensor, predicate, None, "classification test").unwrap(),
                expected
            );
        }
        let tensor = Tensor::from_f32(vec![-f32::MIN_POSITIVE], vec![1, 1]).unwrap();
        assert_eq!(
            classify_tensor(
                &tensor,
                RealUnaryPredicate::Nonnegative,
                None,
                "classification test"
            )
            .unwrap(),
            RealUnaryClassification::Outside
        );
    }
}
