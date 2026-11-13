//! Selected real-domain fusion admission shared by inverse unary owners.
use crate::builtins::common::spec::{
    FusionAdmissionContext, FusionAdmissionEvidence, FusionAdmissionFuture,
};
use runmat_accelerate_api::NumericalDomainEvidence;
use runmat_types::{
    NumericClass, NumericDomain, NumericFact, RealUnaryClassification, RealUnaryPredicate,
    ValueKindFact,
};

pub(crate) fn admit_real_unary(
    context: FusionAdmissionContext<'_>,
    predicate: RealUnaryPredicate,
) -> FusionAdmissionFuture<'_> {
    Box::pin(async move {
        let [fact] = context.facts else {
            return Ok(None);
        };
        let Some(evidence) = context.evidence_at(0) else {
            return Ok(None);
        };
        let required = NumericalDomainEvidence::RealUnary {
            predicate,
            classification: RealUnaryClassification::Inside,
        };
        if evidence.numerical_domain != required {
            return Ok(None);
        }
        let Some(numeric) = fact.numeric() else {
            return Ok(None);
        };
        if numeric.domain != NumericDomain::Real
            || !matches!(numeric.class, NumericClass::Single | NumericClass::Double)
        {
            return Ok(None);
        }
        Ok(Some(FusionAdmissionEvidence {
            output_storage: None,
            output_kind: ValueKindFact::Numeric(NumericFact {
                class: numeric.class,
                domain: NumericDomain::Real,
            }),
            numerical_domain: required,
            arithmetic: None,
            sign: None,
        }))
    })
}
