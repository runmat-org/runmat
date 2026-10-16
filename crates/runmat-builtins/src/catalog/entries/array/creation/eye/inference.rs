use crate::catalog::inference::{default_double_scalar, finish_fixed, literal_text, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, ShapeFact, StorageFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let literals = &request.literals.literal_args;
    let like = literals.iter().position(|literal| {
        literal_text(literal).is_some_and(|text| text.eq_ignore_ascii_case("like"))
    });
    let mut output = like
        .and_then(|index| request.arguments.get(index + 1))
        .cloned()
        .unwrap_or_else(default_double_scalar);
    let last_class = if like.is_none() {
        literals.last().and_then(literal_text)
    } else {
        None
    };
    let dimension_end = like.unwrap_or_else(|| literals.len() - usize::from(last_class.is_some()));
    if let Some(class) = last_class {
        output.kind = if class.eq_ignore_ascii_case("logical") {
            ValueKindFact::Logical
        } else if let Some(class) = NumericClass::from_class_name(&class) {
            numeric_kind(class, NumericDomain::Real)
        } else {
            ValueKindFact::Unknown
        };
    }
    output.shape =
        super::super::shape_facts::dimensions(request, 0, dimension_end, ShapeFact::Scalar);
    output.storage = if matches!(output.storage, StorageFact::Sparse) {
        StorageFact::Sparse
    } else {
        match output.shape.element_count() {
            Some(1)
                if matches!(
                    output.kind,
                    ValueKindFact::Numeric(runmat_types::NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    })
                ) && matches!(output.residency, runmat_types::ResidencyFact::Host) =>
            {
                StorageFact::Scalar
            }
            Some(_) => StorageFact::Dense,
            None => StorageFact::Unknown,
        }
    };
    finish_fixed(entry, request, output, Vec::new())
}
