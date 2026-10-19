use super::policy::DistributionKind;
use runmat_types::*;

fn numeric(class: NumericClass, dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

pub(super) fn request(kind: DistributionKind) -> CallRequest {
    let mut values = vec![
        numeric(NumericClass::Single, &[2, 1]),
        numeric(NumericClass::Double, &[1, 3]),
    ];
    let count = match kind {
        DistributionKind::NormalPdf
        | DistributionKind::NormalCdf
        | DistributionKind::NormalInv
        | DistributionKind::BinomialCdf
        | DistributionKind::WeibullInv => 3,
        DistributionKind::StudentTPdf
        | DistributionKind::StudentTCdf
        | DistributionKind::StudentTInv
        | DistributionKind::ChiSquareCdf => 2,
        DistributionKind::GenericInv => 3,
    };
    if kind == DistributionKind::ChiSquareCdf {
        values[1] = numeric(NumericClass::Double, &[2, 1]);
    }
    if count == 3 {
        values.push(ValueFact::scalar(ValueKindFact::Logical));
    }
    let mut literals = vec![LiteralValue::Unknown; count];
    if kind == DistributionKind::GenericInv {
        values.insert(0, ValueFact::scalar(ValueKindFact::String));
        literals.insert(0, LiteralValue::String("Normal".into()));
    }
    CallRequest {
        arguments: values,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}
