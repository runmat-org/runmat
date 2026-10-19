use crate::catalog::inference::{finish_fixed, literal_text, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    ShapeFact, StorageFact, ValueFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let typename = request.literals.literal_args.last().and_then(literal_text);
    let data_end = request
        .arguments
        .len()
        .saturating_sub(usize::from(typename.is_some()));
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = numeric_kind(
        if typename
            .as_deref()
            .is_some_and(|name| name.eq_ignore_ascii_case("single"))
        {
            NumericClass::Single
        } else {
            NumericClass::Double
        },
        NumericDomain::Real,
    );
    output.shape = if data_end == 1 {
        request
            .arguments
            .first()
            .map(|input| input.shape.clone())
            .unwrap_or(ShapeFact::Unknown)
    } else if data_end >= 3 {
        ShapeFact::from(vec![
            super::super::shape_facts::scalar_dimension(request, 0),
            super::super::shape_facts::scalar_dimension(request, 1),
        ])
    } else {
        ShapeFact::Unknown
    };
    output.storage = StorageFact::Sparse;
    output.residency = ResidencyFact::Host;
    finish_fixed(entry, request, output, Vec::new())
}
