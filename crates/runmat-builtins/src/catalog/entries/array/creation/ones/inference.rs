use crate::catalog::inference::{default_double_scalar, finish_fixed, literal_text, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    ShapeFact, StorageFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let literals = &request.literals.literal_args;
    let like = literals.iter().position(|literal| {
        literal_text(literal).is_some_and(|text| text.eq_ignore_ascii_case("like"))
    });
    let trailing = literals.last().and_then(literal_text);
    let explicit_gpu = trailing
        .as_deref()
        .is_some_and(|text| text.eq_ignore_ascii_case("gpuarray"));
    let class_control = literals
        .get(literals.len().saturating_sub(1 + usize::from(explicit_gpu)))
        .and_then(literal_text)
        .filter(|_| like.is_none());
    let dimension_end = like.unwrap_or_else(|| {
        literals.len().saturating_sub(
            usize::from(trailing.is_some()) + usize::from(explicit_gpu && class_control.is_some()),
        )
    });

    let mut output = like
        .and_then(|index| request.arguments.get(index + 1))
        .cloned()
        .unwrap_or_else(default_double_scalar);
    if let Some(class) = class_control.as_deref() {
        output.kind = if class.eq_ignore_ascii_case("logical") {
            ValueKindFact::Logical
        } else if let Some(class) = NumericClass::from_class_name(class) {
            numeric_kind(class, NumericDomain::Real)
        } else {
            ValueKindFact::Unknown
        };
    }
    output.shape =
        super::super::shape_facts::dimensions(request, 0, dimension_end, ShapeFact::Scalar);
    let prototype_storage = like
        .and_then(|index| request.arguments.get(index + 1))
        .map(|prototype| prototype.storage);
    output.storage = if prototype_storage == Some(StorageFact::Sparse) {
        StorageFact::Sparse
    } else if matches!(
        prototype_storage,
        Some(StorageFact::Unknown | StorageFact::Opaque)
    ) {
        StorageFact::Unknown
    } else {
        match output.shape.element_count() {
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
        }
    };
    if explicit_gpu {
        output.residency = ResidencyFact::Device { provider: None };
        output.storage = StorageFact::Dense;
        if class_control.is_none() {
            output.kind = ValueKindFact::Unknown;
            output.certainty = runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
        }
    }
    if like.is_some() && matches!(output.residency, ResidencyFact::Unknown) {
        output.storage = StorageFact::Unknown;
    }
    finish_fixed(entry, request, output, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, NumericFact, OutputSelection, ValueFact};

    #[test]
    fn like_prototype_selects_representation_not_implicit_size() {
        let mut prototype = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(4), Some(7)]),
            StorageFact::Sparse,
        );
        prototype.residency = ResidencyFact::Host;
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String), prototype],
            literals: LiteralContext::new(vec![
                LiteralValue::String("like".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::ONES_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(result.outputs[0].storage, StorageFact::Sparse);
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            })
        );

        let mut unknown_storage = request;
        unknown_storage.arguments[1].storage = StorageFact::Unknown;
        let result = infer(&unknown_storage, &super::super::ONES_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
    }

    #[test]
    fn scalar_size_does_not_claim_scalar_storage_for_complex_or_integer_like() {
        for class in [NumericClass::Int64, NumericClass::Double] {
            let domain = if class == NumericClass::Double {
                NumericDomain::Complex
            } else {
                NumericDomain::Real
            };
            let mut prototype = ValueFact::proven(
                numeric_kind(class, domain),
                ShapeFact::Scalar,
                StorageFact::Dense,
            );
            prototype.residency = ResidencyFact::Host;
            let request = CallRequest {
                arguments: vec![ValueFact::scalar(ValueKindFact::String), prototype],
                literals: LiteralContext::new(vec![
                    LiteralValue::String("like".into()),
                    LiteralValue::Unknown,
                ]),
                outputs: OutputSelection::default(),
            };
            let result = infer(&request, &super::super::ONES_CATALOG_ENTRY);
            assert_eq!(result.outputs[0].storage, StorageFact::Dense);
        }
    }
}
