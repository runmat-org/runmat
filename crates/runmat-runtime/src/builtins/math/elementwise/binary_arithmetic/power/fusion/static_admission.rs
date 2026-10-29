//! Static real-domain proof and its measured host/projection work.

use super::{selected_arithmetic, sign, SEMANTICS};
use crate::builtins::common::{broadcast::BroadcastPlan, spec::*, tensor};
use runmat_accelerate_api::{
    NumericalDomainEvidence, PowerResultDomain, ProviderPrecision, RealPowerArithmeticMode,
};
use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};
use runmat_value::{Tensor, Value};
use std::borrow::Cow;

pub(super) fn admit_static(
    context: &FusionAdmissionContext<'_>,
) -> anyhow::Result<Option<(FusionAdmissionEvidence, FusionAdmissionWork)>> {
    let (evidence, work) = static_admission_with_work(context)?;
    Ok(evidence.map(|evidence| (evidence, work)))
}

/// A failed real-domain proof still consumed its host scan. Static callers
/// need only successful evidence; observed async admission also retains work
/// from a rejected proof so placement never mistakes it for no preparation.
pub(super) fn static_admission_with_work(
    context: &FusionAdmissionContext<'_>,
) -> anyhow::Result<(Option<FusionAdmissionEvidence>, FusionAdmissionWork)> {
    let Some((receipt, scalar)) = selected_arithmetic(context) else {
        return Ok((None, FusionAdmissionWork::None));
    };
    admission_for_selected(context, receipt, scalar)
}

/// Both static and async entrypoints use this exact theorem after their
/// private selected-arithmetic validation, without allocating its facts twice.
pub(super) fn admission_for_selected(
    context: &FusionAdmissionContext<'_>,
    receipt: FusionPowerArithmeticReceipt,
    scalar: Option<f64>,
) -> anyhow::Result<(Option<FusionAdmissionEvidence>, FusionAdmissionWork)> {
    let [left, Some(right)] = context.operands else {
        return Ok((None, FusionAdmissionWork::None));
    };
    let mode = receipt.mode;
    let precision = if receipt.base_class == NumericClass::Single {
        ProviderPrecision::F32
    } else {
        ProviderPrecision::F64
    };
    let prior_nonnegative = context.evidence_at(0).is_some_and(|evidence| {
        evidence.numerical_domain == NumericalDomainEvidence::NonnegativeOrNaN
    });
    let universally_real = context.facts[1].shape.element_count() == Some(1)
        && scalar.is_some_and(|value| {
            value.is_finite()
                && (super::super::domain::real_for_any_real_base(value) || prior_nonnegative)
        });
    // The represented scalar theorem has already completed the proof. Do
    // not construct a scalar Tensor merely to inspect a disjoint fallback.
    let exponent_host = if universally_real {
        None
    } else {
        host_tensor(right)?
    };
    let host_universally_real = exponent_host.as_ref().is_some_and(|tensor| {
        context.facts[1].shape.element_count() == Some(tensor.len())
            && (0..tensor.len()).all(|index| {
                let value = tensor::tensor_value_f64(tensor, index);
                value.is_finite()
                    && (super::super::domain::real_for_any_real_base(value) || prior_nonnegative)
            })
    });
    let (domain, work) = if universally_real {
        (
            PowerResultDomain::Real,
            FusionAdmissionWork::DomainProof {
                binding: SEMANTICS.binding,
                provider_precision: precision,
                stages: vec![FusionProofStage::RangePropagation { operands: 2 }],
            },
        )
    } else if host_universally_real {
        (
            PowerResultDomain::Real,
            FusionAdmissionWork::Host {
                elements: u64::try_from(exponent_host.as_ref().expect("host proof").len())?,
            },
        )
    } else if let Some(left) = left {
        let (Some(left), Some(right)) = (host_tensor(left)?, host_tensor(right)?) else {
            return Ok((None, FusionAdmissionWork::None));
        };
        let plan = BroadcastPlan::new(left.shape(), right.shape()).map_err(anyhow::Error::msg)?;
        let value = |tensor: &Tensor, index| {
            let value = tensor::tensor_value_f64(tensor, index);
            match mode {
                RealPowerArithmeticMode::NativeSingle => f64::from(value as f32),
                RealPowerArithmeticMode::NativeDouble
                | RealPowerArithmeticMode::MixedDoubleThenSingle { .. } => value,
            }
        };
        (
            super::super::domain::classify_pair(
                &plan,
                |index| value(&left, index),
                |index| value(&right, index),
            ),
            FusionAdmissionWork::Host {
                elements: u64::try_from(plan.len())?,
            },
        )
    } else {
        return Ok((None, FusionAdmissionWork::None));
    };
    if domain != PowerResultDomain::Real {
        return Ok((None, work));
    }
    Ok((
        Some(FusionAdmissionEvidence {
            output_storage: None,
            output_kind: ValueKindFact::Numeric(NumericFact {
                class: mode.output_class(),
                domain: NumericDomain::Real,
            }),
            numerical_domain: NumericalDomainEvidence::RealPower(domain),
            arithmetic: Some(FusionArithmeticEvidence::RealPower(receipt)),
            sign: receipt
                .exponent_scalar
                .filter(|control| sign::even_integral_control(*control))
                .map(|_| FusionSignEvidence::NonnegativeOrNaN {
                    class: mode.output_class(),
                }),
        }),
        work,
    ))
}

fn host_tensor(value: &Value) -> anyhow::Result<Option<Cow<'_, Tensor>>> {
    Ok(match value {
        Value::Num(value) => Some(Cow::Owned(
            Tensor::new(vec![*value], vec![1, 1]).map_err(anyhow::Error::msg)?,
        )),
        Value::Tensor(tensor) if tensor.integer_storage().is_none() => Some(Cow::Borrowed(tensor)),
        _ => None,
    })
}
