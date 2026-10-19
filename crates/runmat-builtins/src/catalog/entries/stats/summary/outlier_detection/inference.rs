//! Static output facts shared by outlier detection and replacement.
use super::{OutlierKind, OutlierOption};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: OutlierKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let input = request.arguments.first();
    if input.is_none() {
        diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "outlier detection requires numeric input data",
        ));
    }
    if kind == OutlierKind::Fill && request.arguments.len() < 2 {
        diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "filloutliers requires a fill method",
        ));
    }
    let mut index = if kind == OutlierKind::Fill { 2 } else { 1 };
    while let Some(literal) = request.literals.literal_args.get(index) {
        let text = match literal {
            LiteralValue::Character(text)
            | LiteralValue::String(text)
            | LiteralValue::Keyword(text) => text,
            _ => {
                index += 1;
                continue;
            }
        };
        match OutlierOption::parse(text) {
            Some(option) => {
                if option.unsupported_array_form()
                    || (kind == OutlierKind::Detect && option == OutlierOption::OutlierLocations)
                {
                    diagnostics.push(InferenceDiagnostic::error(
                        "OPTION",
                        "unsupported numeric-array outlier option",
                    ));
                }
                index += if option.consumes_value() { 2 } else { 1 };
            }
            None => {
                diagnostics.push(InferenceDiagnostic::error(
                    "OPTION",
                    "unsupported outlier method or option",
                ));
                index += 1;
            }
        }
    }
    let class = input.and_then(|value| match value.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => Some(match class {
            NumericClass::Single => NumericClass::Single,
            NumericClass::Double
            | NumericClass::Int8
            | NumericClass::UInt8
            | NumericClass::Int16
            | NumericClass::UInt16
            | NumericClass::Int32
            | NumericClass::UInt32
            | NumericClass::Int64
            | NumericClass::UInt64 => NumericClass::Double,
        }),
        ValueKindFact::Unknown => None,
        _ => {
            diagnostics.push(InferenceDiagnostic::error(
                "INPUT",
                "outlier detection requires real numeric input",
            ));
            None
        }
    });
    let shape = input.map_or(ShapeFact::Unknown, |value| value.shape.clone());
    let mut mask = ValueFact::proven(ValueKindFact::Logical, shape.clone(), StorageFact::Dense);
    mask.residency = match (kind, input.map(|value| &value.residency)) {
        (OutlierKind::Detect, Some(ResidencyFact::Host)) => ResidencyFact::Host,
        (OutlierKind::Detect, _) => ResidencyFact::Unknown,
        (OutlierKind::Fill, _) => ResidencyFact::Host,
    };
    let threshold = || {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Unknown,
            StorageFact::Unknown,
        )
    };
    let mut outputs = match kind {
        OutlierKind::Detect => vec![mask, threshold(), threshold(), threshold()],
        OutlierKind::Fill => {
            let mut filled = match class {
                Some(class) => ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class,
                        domain: NumericDomain::Real,
                    }),
                    shape.clone(),
                    match (class, shape.element_count()) {
                        (NumericClass::Single, _) => StorageFact::Dense,
                        (_, Some(1)) => StorageFact::Scalar,
                        _ => StorageFact::Dense,
                    },
                ),
                None => {
                    let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
                    value.shape = shape.clone();
                    value
                }
            };
            filled.residency = ResidencyFact::Host;
            vec![filled, mask, threshold(), threshold(), threshold()]
        }
    };
    if input.is_none() {
        for output in &mut outputs {
            output.shape = ShapeFact::Unknown;
        }
    }
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount};

    #[test]
    fn mask_is_dense_and_fill_retains_single_or_converts_integer() {
        for (input_class, output_class) in [
            (NumericClass::Single, NumericClass::Single),
            (NumericClass::UInt64, NumericClass::Double),
        ] {
            let request = CallRequest {
                arguments: vec![
                    ValueFact::proven(
                        ValueKindFact::Numeric(NumericFact {
                            class: input_class,
                            domain: NumericDomain::Real,
                        }),
                        ShapeFact::Scalar,
                        StorageFact::Dense,
                    ),
                    ValueFact::unknown(DynamicReason::RuntimeValue),
                ],
                literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            };
            let result = infer(
                &request,
                &super::super::FILLOUTLIERS_CATALOG_ENTRY,
                OutlierKind::Fill,
            );
            assert_eq!(result.outputs[0].numeric().unwrap().class, output_class);
            assert_eq!(result.outputs[1].kind, ValueKindFact::Logical);
            assert_eq!(result.outputs[1].storage, StorageFact::Dense);
        }
    }

    #[test]
    fn symbolic_huge_rank_is_not_materialized() {
        let shape = ShapeFact::Ranked { rank: usize::MAX };
        let request = CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                shape.clone(),
                StorageFact::Dense,
            )],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let result = infer(
            &request,
            &super::super::ISOUTLIER_CATALOG_ENTRY,
            OutlierKind::Detect,
        );
        assert_eq!(result.outputs[0].shape, shape);
    }

    #[test]
    fn unsupported_literal_option_is_diagnosed_without_string_semantic_dispatch() {
        let request = CallRequest {
            arguments: vec![
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }),
                    ShapeFact::Scalar,
                    StorageFact::Dense,
                ),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("SamplePoints".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let result = infer(
            &request,
            &super::super::ISOUTLIER_CATALOG_ENTRY,
            OutlierKind::Detect,
        );
        assert!(!result.diagnostics.is_empty());
    }
}
