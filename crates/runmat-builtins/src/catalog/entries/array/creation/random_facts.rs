use crate::catalog::inference::{
    argument_error, default_double_scalar, finish_fixed, literal_text, numeric_kind,
};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    ShapeFact, StorageFact, ValueKindFact,
};

#[derive(Clone, Copy)]
pub(super) enum RandomConstructor {
    Uniform,
    Normal,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    constructor: RandomConstructor,
) -> CallInference {
    let literals = &request.literals.literal_args;
    let mut diagnostics = Vec::new();
    if matches!(constructor, RandomConstructor::Uniform)
        && literals
            .first()
            .and_then(literal_text)
            .is_some_and(|text| text.eq_ignore_ascii_case("seed"))
    {
        let mut output = default_double_scalar();
        if request.arguments.len() == 2 {
            output.shape = ShapeFact::from(vec![Some(0), Some(0)]);
            output.storage = StorageFact::Dense;
        } else if request.arguments.len() != 1 {
            diagnostics.push(argument_error(
                "RM-RAND-SEED-ARITY",
                "legacy seed control takes zero or one seed value",
                0,
            ));
        }
        return finish_fixed(entry, request, output, diagnostics);
    }
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
    let requested_class = if explicit_gpu {
        literals
            .get(literals.len().saturating_sub(2))
            .and_then(literal_text)
    } else {
        trailing
    };
    let dimension_end = like.unwrap_or_else(|| {
        literals
            .len()
            .saturating_sub(usize::from(requested_class.is_some()) + usize::from(explicit_gpu))
    });
    let mut output = like
        .and_then(|index| request.arguments.get(index + 1))
        .cloned()
        .unwrap_or_else(default_double_scalar);
    if let Some(class) = requested_class.as_deref() {
        output.kind = match NumericClass::from_class_name(class) {
            Some(class @ (NumericClass::Double | NumericClass::Single)) => {
                numeric_kind(class, NumericDomain::Real)
            }
            _ => ValueKindFact::Unknown,
        };
    }
    if let Some(index) = like {
        if let Some(prototype) = request.arguments.get(index + 1) {
            if let Some(numeric) = prototype.numeric() {
                if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                    diagnostics.push(argument_error(
                        "RM-RANDOM-LIKE-CLASS",
                        "random like prototype must be single or double",
                        index + 1,
                    ));
                }
            }
        }
    }
    output.shape = super::shape_facts::dimensions(request, 0, dimension_end, ShapeFact::Scalar);
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
        if requested_class.is_none() {
            output.kind = ValueKindFact::Unknown;
            output.certainty = runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}
