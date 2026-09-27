//! Fusion admission belongs to the same exact paired-domain authority as the
//! public operation. Unknown computed operands cannot borrow ancestor evidence.
use crate::builtins::common::{broadcast::BroadcastPlan, gpu_helpers, spec::*, tensor};
use runmat_accelerate_api::{NumericalDomainEvidence, PowerResultDomain, ProviderPrecision};
use runmat_builtins::{BuiltinBindingIdentity, POWER_CATALOG_ENTRY};
use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};
use runmat_value::{Tensor, Value};
use std::borrow::Cow;

pub(super) const SEMANTICS: FusionSemanticDeclaration = FusionSemanticDeclaration {
    binding: BuiltinBindingIdentity {
        builtin: POWER_CATALOG_ENTRY.identity,
        variant: "default",
    },
    revision: 3,
    admit: Some(admit),
    helpers: &[FusionMathHelper::RealPower],
};

fn admit(context: FusionAdmissionContext<'_>) -> FusionAdmissionFuture<'_> {
    Box::pin(async move {
        let ([Some(left), Some(right)], [left_fact, right_fact]) =
            (context.operands, context.facts)
        else {
            return Ok(None);
        };
        let (Some(left_numeric), Some(right_numeric)) = (left_fact.numeric(), right_fact.numeric())
        else {
            return Ok(None);
        };
        if left_numeric != right_numeric
            || left_numeric.domain != NumericDomain::Real
            || !matches!(
                left_numeric.class,
                NumericClass::Single | NumericClass::Double
            )
        {
            return Ok(None);
        }
        let precision = if left_numeric.class == NumericClass::Single {
            ProviderPrecision::F32
        } else {
            ProviderPrecision::F64
        };
        let domain = if let (Value::GpuTensor(left), Value::GpuTensor(right)) = (left, right) {
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
            let left_metadata = gpu_helpers::HandleMetadataRestoreGuard::new(left);
            let right_metadata = gpu_helpers::HandleMetadataRestoreGuard::new(right);
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
            let (Some(left), Some(right)) = (host_tensor(left)?, host_tensor(right)?) else {
                return Ok(None);
            };
            let plan = BroadcastPlan::new(&left.shape, &right.shape).map_err(anyhow::Error::msg)?;
            *context.work = FusionAdmissionWork::Host {
                elements: u64::try_from(plan.len())?,
            };
            let value = |tensor: &Tensor, index| {
                let value = tensor::tensor_value_f64(tensor, index);
                match precision {
                    ProviderPrecision::F64 => value,
                    ProviderPrecision::F32 => f64::from(value as f32),
                }
            };
            super::domain::classify_pair(
                &plan,
                |index| value(&left, index),
                |index| value(&right, index),
            )
        };
        if domain != PowerResultDomain::Real {
            return Ok(None);
        }
        Ok(Some(FusionAdmissionEvidence {
            output_kind: ValueKindFact::Numeric(NumericFact {
                class: left_numeric.class,
                domain: NumericDomain::Real,
            }),
            numerical_domain: NumericalDomainEvidence::RealPower(domain),
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
