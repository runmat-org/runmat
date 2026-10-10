//! Fusion admission belongs to the same exact paired-domain authority as the
//! public operation. Unknown computed operands cannot borrow ancestor evidence.
use crate::builtins::common::{broadcast::BroadcastPlan, gpu_helpers, spec::*, tensor};
use runmat_accelerate_api::{
    NumericalDomainEvidence, PowerResultDomain, ProviderPrecision, RealPowerArithmeticMode,
};
use runmat_builtins::{BuiltinBindingIdentity, POWER_CATALOG_ENTRY};
use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};
use runmat_value::{NumericScalar, Tensor, Value};
use std::borrow::Cow;

#[path = "fusion/sign.rs"]
mod sign;

pub(super) const SEMANTICS: FusionSemanticDeclaration = FusionSemanticDeclaration {
    binding: BuiltinBindingIdentity {
        builtin: POWER_CATALOG_ENTRY.identity,
        variant: "default",
    },
    revision: 6,
    admit: Some(admit),
    admit_static: Some(admit_static),
    required_real_unary: None,
    required_arithmetic: Some(FusionArithmeticKind::RealPower),
    helpers: &[FusionMathHelper::RealPower],
    scalar_operation: None,
};

fn selected_arithmetic(
    context: &FusionAdmissionContext<'_>,
) -> Option<(FusionPowerArithmeticReceipt, Option<f64>)> {
    let ([left, Some(right)], [left_fact, right_fact]) = (context.operands, context.facts) else {
        return None;
    };
    let (left_numeric, right_numeric) = (left_fact.numeric()?, right_fact.numeric()?);
    if left_numeric.domain != NumericDomain::Real || right_numeric.domain != NumericDomain::Real {
        return None;
    }
    if [*left, Some(*right)]
        .into_iter()
        .zip([left_numeric.class, right_numeric.class])
        .any(|(value, class)| {
            value.is_some_and(|value| {
                crate::value_fact::value_fact(value)
                    .numeric()
                    .is_none_or(|numeric| numeric.class != class)
            })
        })
    {
        return None;
    }
    let mode = RealPowerArithmeticMode::for_classes(left_numeric.class, right_numeric.class)?;
    let (exponent_scalar, scalar) = match right {
        Value::Num(value) => (
            Some(FusionScalarControl::DoubleBits(value.to_bits())),
            Some(*value),
        ),
        Value::Tensor(tensor) if tensor.len() == 1 => match tensor.numeric_value_at(0) {
            Some(NumericScalar::F32(value)) => (
                Some(FusionScalarControl::SingleBits(value.to_bits())),
                Some(f64::from(value)),
            ),
            Some(NumericScalar::F64(value)) => (
                Some(FusionScalarControl::DoubleBits(value.to_bits())),
                Some(value),
            ),
            _ => (None, None),
        },
        _ => (None, None),
    };
    Some((
        FusionPowerArithmeticReceipt {
            binding: SEMANTICS.binding,
            revision: SEMANTICS.revision,
            base_id: context.operand_id(0)?,
            exponent_id: context.operand_id(1)?,
            base_class: left_numeric.class,
            exponent_class: right_numeric.class,
            exponent_scalar,
            mode,
            output_class: mode.output_class(),
        },
        scalar,
    ))
}

fn admit_static(
    context: &FusionAdmissionContext<'_>,
) -> anyhow::Result<Option<(FusionAdmissionEvidence, FusionAdmissionWork)>> {
    let (evidence, work) = static_admission_with_work(context)?;
    Ok(evidence.map(|evidence| (evidence, work)))
}

/// A failed real-domain proof still consumed its host scan. Static callers
/// need only successful evidence; observed async admission also retains work
/// from a rejected proof so placement never mistakes it for no preparation.
fn static_admission_with_work(
    context: &FusionAdmissionContext<'_>,
) -> anyhow::Result<(Option<FusionAdmissionEvidence>, FusionAdmissionWork)> {
    let Some((receipt, scalar)) = selected_arithmetic(context) else {
        return Ok((None, FusionAdmissionWork::None));
    };
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
            value.is_finite() && (super::domain::real_for_any_real_base(value) || prior_nonnegative)
        });
    let exponent_host = host_tensor(right)?;
    let host_universally_real = exponent_host.as_ref().is_some_and(|tensor| {
        context.facts[1].shape.element_count() == Some(tensor.len())
            && (0..tensor.len()).all(|index| {
                let value = tensor::tensor_value_f64(tensor, index);
                value.is_finite()
                    && (super::domain::real_for_any_real_base(value) || prior_nonnegative)
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
            super::domain::classify_pair(
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

fn admit(context: FusionAdmissionContext<'_>) -> FusionAdmissionFuture<'_> {
    Box::pin(async move {
        let Some((receipt, _)) = selected_arithmetic(&context) else {
            return Ok(None);
        };
        let (evidence, work) = static_admission_with_work(&context)?;
        *context.work = work;
        if let Some(evidence) = evidence {
            return Ok(Some(evidence));
        }
        let ([left, Some(right)], _) = (context.operands, context.facts) else {
            return Ok(None);
        };
        let mode = receipt.mode;
        let precision = if receipt.base_class == NumericClass::Single {
            ProviderPrecision::F32
        } else {
            ProviderPrecision::F64
        };
        // The ordinary mixed Single/Double route evaluates in binary64 before
        // rounding its result to Single. The current fused F32 helper cannot
        // claim that arithmetic mode, even when the scalar exponent converts
        // exactly to F32 (near-rounding-boundary results can still differ).
        if left.is_none() {
            return Ok(None);
        }
        let Some(left) = left else {
            return Ok(None);
        };
        let domain = if let (Value::GpuTensor(left), Value::GpuTensor(right)) = (left, right) {
            if matches!(mode, RealPowerArithmeticMode::MixedDoubleThenSingle { .. }) {
                // Paired resident classification currently admits one native
                // precision. A mixed physical route needs its own probe.
                return Ok(None);
            }
            // Classification is about the exact values that the kernel will
            // receive, not a future lossy conversion of different operands.
            if [left, right]
                .iter()
                .any(|handle| runmat_accelerate_api::handle_precision(handle) != Some(precision))
            {
                return Ok(None);
            }
            let provider = gpu_helpers::exact_provider_for_binary_inputs(left, right)
                .map_err(anyhow::Error::msg)?;
            *context.work = FusionAdmissionWork::ProviderProbe {
                binding: SEMANTICS.binding,
                precision,
                provider_precision: provider.precision(),
                inputs: [left, right]
                    .into_iter()
                    .map(super::provider::admission::representation)
                    .collect::<Result<Vec<_>, _>>()?,
            };
            let left_metadata = gpu_helpers::ResidencyRestoreGuard::new(left);
            let right_metadata = gpu_helpers::ResidencyRestoreGuard::new(right);
            let result = provider.classify_real_power(left, right).await;
            drop((left_metadata, right_metadata));
            match result {
                Ok(domain) => domain,
                Err(error) if runmat_accelerate_api::is_unsupported_provider_operation(&error) => {
                    return Ok(None)
                }
                Err(error) => return Err(error),
            }
        } else {
            return Ok(None);
        };
        if domain != PowerResultDomain::Real {
            return Ok(None);
        }
        Ok(Some(FusionAdmissionEvidence {
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
        }))
    })
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

#[cfg(test)]
#[path = "fusion/tests.rs"]
mod tests;
