use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-REGEXPREP-ARITY",
            "regexprep requires subject, pattern and replacement",
            request.arguments.len(),
        ));
    }
    for (index, argument) in request.arguments.iter().take(3).enumerate() {
        if !matches!(
            &argument.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-REGEXPREP-TEXT",
                "regexprep requires host text in subject, pattern and replacement roles",
                index,
            ));
        }
    }
    super::options::diagnose(request, &mut diagnostics);
    let output = match request.arguments.first() {
        Some(subject) => match &subject.kind {
            ValueKindFact::String | ValueKindFact::Cell(_) => {
                let mut output = subject.clone();
                output.alias = runmat_types::AliasFact::Unique;
                output.residency = runmat_types::ResidencyFact::Host;
                output
            }
            ValueKindFact::Character => {
                let shape = subject
                    .shape
                    .known_dims()
                    .map_or(ShapeFact::Unknown, |dims| {
                        ShapeFact::from(vec![dims.first().copied().flatten(), None])
                    });
                let mut output =
                    ValueFact::proven(ValueKindFact::Character, shape, StorageFact::Dense);
                output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
                output
            }
            _ => ValueFact::unknown(DynamicReason::RuntimeValue),
        },
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn string_array_shape_survives_and_char_width_does_not() {
        let entry = &super::super::REGEXPREP_CATALOG_ENTRY;
        let mut request = CallRequest {
            arguments: vec![
                ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(vec![Some(2), Some(3)]),
                    StorageFact::Dense,
                ),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            infer(&request, entry).outputs[0].shape,
            request.arguments[0].shape
        );
        request.arguments[0] = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(8)]),
            StorageFact::Dense,
        );
        assert_eq!(
            infer(&request, entry).outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn known_options_and_occurrences_follow_runtime_admission() {
        use runmat_types::{LiteralValue, NumericClass};
        let entry = &super::super::REGEXPREP_CATALOG_ENTRY;
        let base = vec![ValueFact::scalar(ValueKindFact::String); 3];
        let check = |tail: Vec<(ValueFact, LiteralValue)>| {
            let mut arguments = base.clone();
            let mut literals = vec![LiteralValue::Unknown; 3];
            for (fact, literal) in tail {
                arguments.push(fact);
                literals.push(literal);
            }
            infer(
                &CallRequest {
                    arguments,
                    literals: LiteralContext::new(literals),
                    outputs: OutputSelection::default(),
                },
                entry,
            )
            .diagnostics
        };
        let text = || ValueFact::scalar(ValueKindFact::String);
        let numeric = |class| {
            ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
                class,
                domain: runmat_types::NumericDomain::Real,
            }))
        };
        assert!(check(vec![(text(), LiteralValue::String("IgnoreCase".into()))]).is_empty());
        assert!(
            check(vec![(text(), LiteralValue::String("notAnOption".into()))])
                .iter()
                .any(|d| d.code == "RM-CATALOG-REGEXPREP-OPTION")
        );
        for invalid in [LiteralValue::Number(0.0), LiteralValue::Number(1.5)] {
            assert!(check(vec![(numeric(NumericClass::Double), invalid)])
                .iter()
                .any(|d| d.code == "RM-CATALOG-REGEXPREP-OCCURRENCE"));
        }
        assert!(check(vec![(
            numeric(NumericClass::Int64),
            LiteralValue::Integer {
                text: "-1".into(),
                class: NumericClass::Int64,
            }
        )])
        .iter()
        .any(|d| d.code == "RM-CATALOG-REGEXPREP-OCCURRENCE"));
        assert!(check(vec![(
            numeric(NumericClass::UInt64),
            LiteralValue::Integer {
                text: "18446744073709551615".into(),
                class: NumericClass::UInt64,
            }
        )])
        .is_empty());
        assert!(check(vec![
            (text(), LiteralValue::String("dotall".into())),
            (text(), LiteralValue::String("off".into())),
            (text(), LiteralValue::String("once".into())),
        ])
        .is_empty());
        assert!(check(vec![
            (text(), LiteralValue::String("warnings".into())),
            (text(), LiteralValue::String("notAnOption".into())),
        ])
        .is_empty());
        let single = || {
            ValueFact::proven(
                ValueKindFact::Numeric(runmat_types::NumericFact {
                    class: NumericClass::Single,
                    domain: runmat_types::NumericDomain::Real,
                }),
                ShapeFact::Scalar,
                StorageFact::Dense,
            )
        };
        let one_single = || LiteralValue::Real {
            text: "1".into(),
            class: NumericClass::Single,
        };
        assert!(check(vec![
            (text(), LiteralValue::String("preservecase".into())),
            (single(), one_single()),
        ])
        .iter()
        .any(|d| d.code == "RM-CATALOG-REGEXPREP-OPTION"));
        assert!(check(vec![
            (text(), LiteralValue::String("lineanchors".into())),
            (single(), one_single()),
            (text(), LiteralValue::String("notAnOption".into())),
        ])
        .iter()
        .any(|d| d.code == "RM-CATALOG-REGEXPREP-OPTION" && d.argument == Some(5)));
        assert!(check(vec![
            (text(), LiteralValue::String("dotall".into())),
            (
                single(),
                LiteralValue::Real {
                    text: "0".into(),
                    class: NumericClass::Single
                }
            ),
            (text(), LiteralValue::String("once".into())),
        ])
        .iter()
        .any(|d| d.code == "RM-CATALOG-REGEXPREP-OCCURRENCE" && d.argument == Some(4)));
        for flag in ["lineanchors", "preservecase"] {
            assert!(check(vec![
                (text(), LiteralValue::String(flag.into())),
                (
                    ValueFact::unknown(DynamicReason::RuntimeValue),
                    LiteralValue::Unknown
                ),
                (text(), LiteralValue::String("notAnOption".into())),
            ])
            .is_empty());
        }
    }
}
