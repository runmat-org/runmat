//! Semantic-only SQRT refinement. Physical fusion remains disabled.

use crate::builtins::common::spec::{
    FusionAdmission, FusionAdmissionContext, FusionAdmissionEvidence, FusionAdmissionFuture,
    FusionAdmissionWork, FusionProofStage, FusionSemanticDeclaration, FusionSignEvidence,
};
use runmat_accelerate_api::{NumericalDomainEvidence, ProviderPrecision};
use runmat_builtins::{BuiltinBindingIdentity, SQRT_CATALOG_ENTRY};
use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};

pub(super) const SEMANTICS: FusionSemanticDeclaration = FusionSemanticDeclaration {
    binding: BuiltinBindingIdentity {
        builtin: SQRT_CATALOG_ENTRY.identity,
        variant: "default",
    },
    revision: 2,
    admit: Some(FusionAdmission::unbounded(admit)),
    admit_static: Some(admit_static),
    required_real_unary: None,
    required_arithmetic: None,
    helpers: &[],
    scalar_operation: None,
};

fn refine(context: &FusionAdmissionContext<'_>) -> Option<FusionAdmissionEvidence> {
    let [input] = context.facts else {
        return None;
    };
    let output_fact = context.output_fact?;
    let numeric = input.numeric()?;
    let Some(FusionSignEvidence::NonnegativeOrNaN { class }) =
        context.evidence_at(0).and_then(|evidence| evidence.sign)
    else {
        return None;
    };
    if numeric.domain != NumericDomain::Real
        || numeric.class != class
        || !matches!(class, NumericClass::Single | NumericClass::Double)
        || input.shape != output_fact.shape
        || output_fact
            .numeric()
            .is_some_and(|output| output.class != class || output.domain != NumericDomain::Real)
    {
        return None;
    }
    Some(FusionAdmissionEvidence {
        output_storage: super::publication::storage(input, output_fact, class),
        output_kind: ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        numerical_domain: NumericalDomainEvidence::Unknown,
        arithmetic: None,
        sign: Some(FusionSignEvidence::NonnegativeOrNaN { class }),
    })
}

fn unchanged(context: &FusionAdmissionContext<'_>) -> Option<FusionAdmissionEvidence> {
    Some(FusionAdmissionEvidence {
        output_storage: None,
        output_kind: context.output_fact?.kind.clone(),
        numerical_domain: NumericalDomainEvidence::Unknown,
        arithmetic: None,
        sign: None,
    })
}

fn admit_static(
    context: &FusionAdmissionContext<'_>,
) -> anyhow::Result<Option<(FusionAdmissionEvidence, FusionAdmissionWork)>> {
    let Some(evidence) = refine(context) else {
        return Ok(unchanged(context).map(|evidence| (evidence, FusionAdmissionWork::None)));
    };
    let class = evidence.sign.expect("refined sqrt sign").class();
    Ok(Some((
        evidence,
        FusionAdmissionWork::DomainProof {
            binding: SEMANTICS.binding,
            provider_precision: if class == NumericClass::Single {
                ProviderPrecision::F32
            } else {
                ProviderPrecision::F64
            },
            stages: vec![FusionProofStage::RangePropagation { operands: 1 }],
        },
    )))
}

fn admit(context: FusionAdmissionContext<'_>) -> FusionAdmissionFuture<'_> {
    Box::pin(async move {
        let evidence = refine(&context);
        if let Some(sign) = evidence.as_ref().and_then(|evidence| evidence.sign) {
            *context.work = FusionAdmissionWork::DomainProof {
                binding: SEMANTICS.binding,
                provider_precision: if sign.class() == NumericClass::Single {
                    ProviderPrecision::F32
                } else {
                    ProviderPrecision::F64
                },
                stages: vec![FusionProofStage::RangePropagation { operands: 1 }],
            };
        }
        Ok(evidence.or_else(|| unchanged(&context)))
    })
}
