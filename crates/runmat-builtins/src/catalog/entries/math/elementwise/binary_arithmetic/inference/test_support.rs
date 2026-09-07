use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) fn numeric(
    class: NumericClass,
    domain: NumericDomain,
    shape: &[Option<usize>],
) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(shape.to_vec()),
        StorageFact::Dense,
    )
}

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) fn numeric_scalar(
    class: NumericClass,
    domain: NumericDomain,
) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    )
}

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) fn request(
    arguments: Vec<ValueFact>,
) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}
