use crate::catalog::inference::{
    argument_error, default_double_scalar, finish_fixed, literal_text, numeric_kind,
};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    ShapeFact, StorageFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-RANDI-ARITY",
            "randi requires an upper bound or two-element bounds vector",
            0,
        ));
    }
    let literals = &request.literals.literal_args;
    let like = literals.iter().position(|literal| {
        literal_text(literal).is_some_and(|text| text.eq_ignore_ascii_case("like"))
    });
    let trailing = if like.is_none() {
        literals.last().and_then(literal_text)
    } else {
        None
    };
    let explicit_gpu = trailing
        .as_deref()
        .is_some_and(|text| text.eq_ignore_ascii_case("gpuarray"));
    let class = if explicit_gpu {
        literals
            .get(literals.len().saturating_sub(2))
            .and_then(literal_text)
    } else {
        trailing
    };
    let end = like.unwrap_or_else(|| {
        literals
            .len()
            .saturating_sub(usize::from(class.is_some()) + usize::from(explicit_gpu))
    });
    let mut output = like
        .and_then(|index| request.arguments.get(index + 1))
        .cloned()
        .unwrap_or_else(default_double_scalar);
    if let Some(class) = class.as_deref() {
        output.kind = if class.eq_ignore_ascii_case("logical") {
            ValueKindFact::Logical
        } else if let Some(class) = NumericClass::from_class_name(class) {
            numeric_kind(class, NumericDomain::Real)
        } else {
            ValueKindFact::Unknown
        };
    }
    output.shape = super::super::shape_facts::dimensions(request, 1, end.max(1), ShapeFact::Scalar);
    output.storage = match output.shape.element_count() {
        Some(1)
            if matches!(
                output.kind,
                ValueKindFact::Numeric(runmat_types::NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })
            ) && output.residency == ResidencyFact::Host =>
        {
            StorageFact::Scalar
        }
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    if explicit_gpu {
        output.residency = ResidencyFact::Device { provider: None };
        output.storage = StorageFact::Dense;
        if class.is_none() {
            output.kind = ValueKindFact::Unknown;
            output.certainty = runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}
