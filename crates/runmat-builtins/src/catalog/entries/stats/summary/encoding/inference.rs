//! Sound static facts for the encoding family. Unknown class lists and
//! resident controls stay dynamic; they are not guessed from argument shape.
use super::{EncodingOptionName, EncodingOutputKind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    InferenceDiagnostic, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn error(code: &str, message: &str) -> InferenceDiagnostic {
    InferenceDiagnostic::error(code, message)
}

fn finish(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    output: ValueFact,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let output = if diagnostics.is_empty() {
        output
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

pub(super) fn dummyvar(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(error("ARITY", "dummyvar requires one grouping argument"));
    }
    let rows = request
        .arguments
        .first()
        .and_then(|value| match (&value.kind, &value.shape) {
            (ValueKindFact::Numeric(_) | ValueKindFact::Logical, ShapeFact::Shaped { dims })
                if dims.len() >= 2 =>
            {
                // Cell shape describes the container, not its grouping rows.
                if dims.first() == Some(&DimensionFact::Known(1)) {
                    dims.get(1).cloned()
                } else {
                    dims.first().cloned()
                }
            }
            _ => None,
        });
    let shape = ShapeFact::Shaped {
        dims: vec![
            rows.unwrap_or(DimensionFact::Unknown),
            DimensionFact::Unknown,
        ],
    };
    finish(
        request,
        entry,
        numeric(NumericClass::Double, shape),
        diagnostics,
    )
}

fn literal_text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::Character(text)
        | LiteralValue::String(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

fn literal_dim(value: Option<&LiteralValue>) -> Result<Option<usize>, ()> {
    match value {
        None | Some(LiteralValue::Unknown) => Ok(None),
        Some(LiteralValue::Integer { text, .. }) => text
            .parse::<usize>()
            .ok()
            .filter(|dim| *dim > 0)
            .map(Some)
            .ok_or(()),
        Some(value) => LiteralContext::numeric_from_literal(value)
            .and_then(|number| runmat_types::decode_shape_dimension(number).ok())
            .filter(|dim| *dim > 0)
            .map(Some)
            .ok_or(()),
    }
}

#[derive(Clone, Copy)]
enum OutputSelection {
    Known(EncodingOutputKind),
    Dynamic,
    Invalid,
}

fn output_selection(request: &CallRequest, start: usize, encode: bool) -> OutputSelection {
    let count = request.arguments.len().saturating_sub(start);
    if count == 0 {
        return OutputSelection::Known(if encode {
            EncodingOutputKind::Numeric(NumericClass::Double)
        } else {
            EncodingOutputKind::Categorical
        });
    }
    let mut idx = 0;
    let mut selected = None;
    if count == 1 || (encode && !count.is_multiple_of(2)) {
        let Some(value) = request.literals.literal_args.get(start) else {
            return OutputSelection::Dynamic;
        };
        let Some(text) = literal_text(value) else {
            return if matches!(value, LiteralValue::Unknown) {
                OutputSelection::Dynamic
            } else {
                OutputSelection::Invalid
            };
        };
        selected = EncodingOutputKind::from_public(text);
        if selected.is_none() {
            return OutputSelection::Invalid;
        }
        idx = 1;
    }
    if !(count - idx).is_multiple_of(2) {
        return OutputSelection::Invalid;
    }
    while idx < count {
        let Some(name) = request.literals.literal_args.get(start + idx) else {
            return OutputSelection::Dynamic;
        };
        let Some(text) = literal_text(name) else {
            return if matches!(name, LiteralValue::Unknown) {
                OutputSelection::Dynamic
            } else {
                OutputSelection::Invalid
            };
        };
        match EncodingOptionName::from_public(text) {
            Some(EncodingOptionName::Classes) if encode => {}
            Some(EncodingOptionName::OutputType) => {
                let Some(value) = request.literals.literal_args.get(start + idx + 1) else {
                    return OutputSelection::Dynamic;
                };
                let Some(text) = literal_text(value) else {
                    return if matches!(value, LiteralValue::Unknown) {
                        OutputSelection::Dynamic
                    } else {
                        OutputSelection::Invalid
                    };
                };
                selected = EncodingOutputKind::from_public(text);
                if selected.is_none() {
                    return OutputSelection::Invalid;
                }
            }
            _ => return OutputSelection::Invalid,
        }
        idx += 2;
    }
    let selected = selected.unwrap_or(if encode {
        EncodingOutputKind::Numeric(NumericClass::Double)
    } else {
        EncodingOutputKind::Categorical
    });
    if encode
        && !matches!(
            selected,
            EncodingOutputKind::Numeric(_) | EncodingOutputKind::Logical
        )
    {
        OutputSelection::Invalid
    } else {
        OutputSelection::Known(selected)
    }
}

fn selected_fact(selection: OutputSelection, shape: ShapeFact) -> ValueFact {
    match selection {
        OutputSelection::Known(EncodingOutputKind::Numeric(class)) => numeric(class, shape),
        OutputSelection::Known(EncodingOutputKind::Logical) => {
            ValueFact::proven(ValueKindFact::Logical, shape, StorageFact::Dense)
        }
        OutputSelection::Known(EncodingOutputKind::String) => {
            ValueFact::proven(ValueKindFact::String, shape, StorageFact::Dense)
        }
        OutputSelection::Known(EncodingOutputKind::Categorical) => ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(runmat_types::standard::CATEGORICAL.owned()),
                properties: Default::default(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Unknown,
            StorageFact::Unknown,
        ),
        OutputSelection::Known(EncodingOutputKind::Cell)
        | OutputSelection::Dynamic
        | OutputSelection::Invalid => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

pub(super) fn encode(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(error(
            "ARITY",
            "onehotencode requires labels and featureDim",
        ));
    }
    let axis = match literal_dim(request.literals.literal_args.get(1)) {
        Ok(axis) => axis,
        Err(()) => {
            diagnostics.push(error(
                "DIMENSION",
                "featureDim must be a positive integer scalar",
            ));
            None
        }
    };
    let selection = output_selection(request, 2, true);
    if matches!(selection, OutputSelection::Invalid) {
        diagnostics.push(error(
            "OPTION",
            "invalid onehotencode output type or option",
        ));
    }
    let shape = match (request.arguments.first(), axis) {
        (Some(input), Some(axis)) => {
            // Character arrays encode one label per row, and categorical
            // objects obtain their geometry from Codes. Their outer fact
            // shape is not the label shape.
            let mut dims = if matches!(
                input.kind,
                ValueKindFact::Numeric(_)
                    | ValueKindFact::Logical
                    | ValueKindFact::String
                    | ValueKindFact::Cell(_)
            ) {
                input.shape.known_dims()
            } else {
                None
            };
            if let Some(ref mut dims) = dims {
                if axis <= 32 {
                    dims.resize(axis.max(dims.len()), Some(1));
                    if dims[axis - 1] != Some(1) {
                        diagnostics.push(error(
                            "DIMENSION",
                            "featureDim must identify a singleton dimension",
                        ));
                    }
                    dims[axis - 1] = None; // ClassNames/category count is value-dependent.
                } else {
                    diagnostics.push(error("DIMENSION", "featureDim is too large"));
                    dims.clear();
                }
            }
            dims.filter(|dims| !dims.is_empty())
                .map_or(ShapeFact::Unknown, ShapeFact::from)
        }
        _ => ShapeFact::Unknown,
    };
    finish(request, entry, selected_fact(selection, shape), diagnostics)
}

pub(super) fn decode(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(error(
            "ARITY",
            "onehotdecode requires encoded values, classes, and featureDim",
        ));
    }
    let axis = match literal_dim(request.literals.literal_args.get(2)) {
        Ok(axis) => axis,
        Err(()) => {
            diagnostics.push(error(
                "DIMENSION",
                "featureDim must be a positive integer scalar",
            ));
            None
        }
    };
    let selection = output_selection(request, 3, false);
    if matches!(selection, OutputSelection::Invalid) {
        diagnostics.push(error(
            "OPTION",
            "invalid onehotdecode output type or option",
        ));
    }
    let shape = match (request.arguments.first(), axis) {
        (Some(input), Some(axis)) => {
            let admitted_shape = if matches!(
                input.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) {
                input.shape.known_dims()
            } else {
                None
            };
            admitted_shape
                .and_then(|mut dims| {
                    if axis > dims.len() {
                        diagnostics.push(error(
                            "DIMENSION",
                            "featureDim exceeds the encoded array rank",
                        ));
                        return None;
                    }
                    dims[axis - 1] = Some(1);
                    Some(ShapeFact::from(dims))
                })
                .unwrap_or(ShapeFact::Unknown)
        }
        _ => ShapeFact::Unknown,
    };
    finish(request, entry, selected_fact(selection, shape), diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{CellFact, OutputSelection as RequestedOutputs, RequestedOutputCount};

    fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: RequestedOutputs::new(RequestedOutputCount::Exactly(1)),
        }
    }

    fn numeric_fact(shape: ShapeFact) -> ValueFact {
        numeric(NumericClass::Double, shape)
    }

    #[test]
    fn encode_preserves_known_axes_without_guessing_class_count() {
        let args = vec![
            ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(vec![Some(3), Some(1)]),
                StorageFact::Dense,
            ),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            ValueFact::scalar(ValueKindFact::Character),
            ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(vec![Some(2), Some(1)]),
                StorageFact::Dense,
            ),
        ];
        let call = request(
            args,
            vec![
                LiteralValue::Unknown,
                LiteralValue::Number(2.0),
                LiteralValue::Character("ClassNames".into()),
                LiteralValue::Unknown,
            ],
        );
        let result = encode(&call, &super::super::ONEHOTENCODE_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(3), None])
        );
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        );
    }

    #[test]
    fn invalid_known_dimension_does_not_publish_output_fact() {
        let call = request(
            vec![
                numeric_fact(ShapeFact::from(vec![Some(3), Some(1)])),
                numeric_fact(ShapeFact::Scalar),
            ],
            vec![LiteralValue::Unknown, LiteralValue::Number(0.0)],
        );
        let result = encode(&call, &super::super::ONEHOTENCODE_CATALOG_ENTRY);
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }

    #[test]
    fn decode_known_geometry_and_typed_output() {
        let args = vec![
            numeric_fact(ShapeFact::from(vec![Some(3), Some(2)])),
            ValueFact::unknown(DynamicReason::RuntimeValue),
            numeric_fact(ShapeFact::Scalar),
            ValueFact::scalar(ValueKindFact::Character),
        ];
        let call = request(
            args,
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Integer {
                    text: "2".into(),
                    class: NumericClass::UInt64,
                },
                LiteralValue::Character("uint64".into()),
            ],
        );
        let result = decode(&call, &super::super::ONEHOTDECODE_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(1)])
        );
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real
            })
        );
    }

    #[test]
    fn character_rows_and_cell_group_containers_do_not_reuse_outer_shape() {
        let char_input = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(3), Some(4)]),
            StorageFact::Dense,
        );
        let dim = numeric_fact(ShapeFact::Scalar);
        let call = request(
            vec![char_input, dim],
            vec![LiteralValue::Unknown, LiteralValue::Number(2.0)],
        );
        let encoded = encode(&call, &super::super::ONEHOTENCODE_CATALOG_ENTRY);
        assert_eq!(encoded.outputs[0].shape, ShapeFact::Unknown);

        let cell_input = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                elements: Vec::new(),
                elements_complete: false,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let call = request(vec![cell_input], vec![LiteralValue::Unknown]);
        let dummy = dummyvar(&call, &super::super::DUMMYVAR_CATALOG_ENTRY);
        assert_eq!(dummy.outputs[0].shape, ShapeFact::from(vec![None, None]));
    }

    #[test]
    fn invalid_known_output_type_remains_dynamic() {
        let call = request(
            vec![
                numeric_fact(ShapeFact::from(vec![Some(3), Some(1)])),
                numeric_fact(ShapeFact::Scalar),
                ValueFact::scalar(ValueKindFact::Character),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Number(2.0),
                LiteralValue::Character("categorical".into()),
            ],
        );
        let result = encode(&call, &super::super::ONEHOTENCODE_CATALOG_ENTRY);
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
}
