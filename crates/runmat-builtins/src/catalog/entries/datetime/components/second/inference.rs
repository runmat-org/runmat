use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-SECOND-ARITY",
            "second requires one datetime input",
            0,
        ));
    }
    let shape = match request.arguments.first() {
        Some(ValueFact {
            kind: ValueKindFact::Object(object),
            shape,
            ..
        }) => match object.runtime_class.as_ref() {
            Some(class) if class == &standard::DATETIME.owned() => shape.clone(),
            Some(_) => {
                diagnostics.push(argument_error(
                    "RM-SECOND-CLASS",
                    "second requires a datetime object",
                    0,
                ));
                ShapeFact::Unknown
            }
            None => ShapeFact::Unknown,
        },
        Some(value) if value.kind == ValueKindFact::Unknown => ShapeFact::Unknown,
        Some(_) => {
            diagnostics.push(argument_error(
                "RM-SECOND-INPUT",
                "second does not convert text, numeric serials, or duration values",
                0,
            ));
            ShapeFact::Unknown
        }
        None => ShapeFact::Unknown,
    };
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    finish_fixed(
        entry,
        request,
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            storage,
        ),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    #[test]
    fn datetime_shape_double_output_and_no_clock_authority() {
        let input = ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(standard::DATETIME.owned()),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::from(vec![Some(2), Some(1), Some(3)]),
            StorageFact::Opaque,
        );
        let mut request = CallRequest {
            arguments: vec![input.clone()],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let entry = &super::super::SECOND_CATALOG_ENTRY;
        let result = infer(&request, entry);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].shape, input.shape);
        assert!(matches!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        ));
        assert!(!entry.contract.effects.contains(&EffectKind::Clock));
        assert_eq!(
            entry.contract.async_behavior,
            crate::BuiltinAsyncBehavior::MaySuspend
        );
        assert!(entry.contract.effects.contains(&EffectKind::MaySuspend));
        request.arguments = vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }))];
        assert!(!infer(&request, entry).diagnostics.is_empty());
        request.arguments = vec![ValueFact::unknown(DynamicReason::RuntimeValue)];
        assert_eq!(infer(&request, entry).outputs[0].shape, ShapeFact::Unknown);
    }
}
