//! Preparation of an admitted real floating operand with request-local domain evidence.
use crate::context::preparation::{
    execute_request_compound, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
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
pub(crate) use runner::RealUnaryAdapter;

pub(crate) async fn execute<F, Fut, B, H, R>(
    runtime: Option<&RuntimeContext>,
    entry: &'static BuiltinCatalogEntry,
    operation: runmat_types::ScalarUnaryMath,
    publication: RealUnaryPublication,
    adapter: RealUnaryAdapter<B, H, R>,
    value: Value,
    body: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
    B: FnOnce(
        &runmat_accelerate_api::GpuTensorHandle,
        RealUnaryClassification,
    ) -> BuiltinResult<()>,
    H: FnOnce(
        &'static dyn runmat_accelerate_api::AccelProvider,
        &runmat_accelerate_api::GpuTensorHandle,
        Value,
    ) -> BuiltinResult<Value>,
    R: FnOnce(
        &'static dyn runmat_accelerate_api::AccelProvider,
        &runmat_accelerate_api::GpuTensorHandle,
        runmat_accelerate_api::GpuTensorHandle,
    ) -> BuiltinResult<Value>,
{
    let predicate = operation.real_predicate().ok_or_else(|| {
        crate::build_runtime_error("real unary preparation requires a promoting-domain instruction")
            .with_identifier("RunMat:Placement:InvalidRequest")
            .build()
    })?;
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    if let Some(route) = crate::builtins::math::host_route::canonical(
        runtime,
        crate::context::VmFusionHostInstruction::ElementwiseCall,
        std::iter::once(&value),
    ) {
        // The admitted host operand reaches the existing host-only numeric
        // body. Domain diagnostics and complex-result semantics stay there;
        // only provider-oriented preparation/promotion is bypassed.
        let evidence = classify_value(&value, predicate);
        let output = body(value).await?;
        route.confirm_numerical_host_branch(
            runtime.unwrap(),
            crate::context::VmFusionHostInstruction::ElementwiseCall,
            evidence,
        );
        return Ok(output);
    }
    let resident = matches!(&value, Value::GpuTensor(handle)
        if crate::builtins::math::host_route::real_floating_shape(&value).is_some());
    if !resident
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
        [Value::Tensor(tensor)] => Ok(NumericalDomainEvidence::RealUnary {
            predicate,
            classification: classify(tensor, predicate),
        }),
        [Value::GpuTensor(_)] if resident => Ok(NumericalDomainEvidence::Unknown),
        _ => unreachable!("admitted floating operand"),
    })?;
    if !resident
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
            .with_owned_values_async(|mut values| body(values.remove(0)))
            .await;
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
    execute_request_compound(runtime, request, |mut values, compound| async move {
        let value = values.remove(0);
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
            return runner::execute(runtime, compound, value, body, adapter).await;
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
        let evidence = if route.is_some() {
            Some(classify_value(&value, predicate))
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
    })
    .await
}

fn classify(tensor: &Tensor, predicate: RealUnaryPredicate) -> RealUnaryClassification {
    classify_values(
        (0..tensor.len())
            .map(|index| crate::builtins::common::tensor::tensor_value_f64(tensor, index)),
        predicate,
    )
}

fn classify_value(value: &Value, predicate: RealUnaryPredicate) -> NumericalDomainEvidence {
    let classification = match value {
        Value::Tensor(tensor) => classify(tensor, predicate),
        Value::Num(value) => classify_values(std::iter::once(*value), predicate),
        _ => unreachable!("canonical native floating host branch"),
    };
    NumericalDomainEvidence::RealUnary {
        predicate,
        classification,
    }
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
            assert_eq!(classify(&tensor, predicate), expected);
        }
        let tensor = Tensor::from_f32(vec![-f32::MIN_POSITIVE], vec![1, 1]).unwrap();
        assert_eq!(
            classify(&tensor, RealUnaryPredicate::Nonnegative),
            RealUnaryClassification::Outside
        );
    }
}
