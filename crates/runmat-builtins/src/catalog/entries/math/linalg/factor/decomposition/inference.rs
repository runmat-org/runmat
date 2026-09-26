//! Domain facts for opaque, value-semantic decomposition objects.
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

fn object() -> ValueFact {
    let mut properties = BTreeMap::new();
    for name in ["Type", "Datatype"] {
        properties.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    for name in [
        "CheckCondition",
        "IsConjugateTransposed",
        "IsReal",
        "IsSparse",
    ] {
        properties.insert(name.into(), ValueFact::scalar(ValueKindFact::Logical));
    }
    properties.insert(
        "MatrixSize".into(),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        ),
    );
    // ScaleFactor can become real or complex; its value is not a class fact.
    let mut scale = ValueFact::unknown(DynamicReason::RuntimeValue);
    scale.shape = ShapeFact::Scalar;
    scale.residency = ResidencyFact::Host;
    properties.insert("ScaleFactor".into(), scale);
    let mut result = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("decomposition").owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    result.residency = ResidencyFact::Host;
    result
}

fn is_object(value: &ValueFact) -> bool {
    matches!(&value.kind, ValueKindFact::Object(fact)
        if fact.runtime_class.as_ref() == Some(&StaticClassIdentity::new("decomposition").owned()))
}

fn arity(request: &CallRequest, count: usize) -> Vec<InferenceDiagnostic> {
    if request.arguments.len() == count {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-DECOMPOSITION-ARITY",
            format!("expected {count} inputs"),
            0,
        )]
    }
}

fn receiver(request: &CallRequest, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if let Some(value) = request.arguments.get(index) {
        if !is_object(value) && !matches!(value.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-DECOMPOSITION-RECEIVER",
                "expected a decomposition object",
                index,
            ));
        }
    }
}

pub(super) fn constructor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(matrix) = request.arguments.first() {
        if !matches!(
            matrix.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) || matrix.storage == StorageFact::Sparse
            || matrix.shape.known_dims().is_some_and(|dims| dims.len() > 2)
        {
            diagnostics.push(argument_error(
                "RM-DECOMPOSITION-MATRIX",
                "expected a dense two-dimensional numeric matrix",
                0,
            ));
        }
        if let Some(kind) = request.literals.literal_string_at(1) {
            if ![
                "auto",
                "lu",
                "qr",
                "chol",
                "triangular",
                "diagonal",
                "checkcondition",
                "ranktolerance",
            ]
            .contains(&kind.as_str())
            {
                diagnostics.push(argument_error(
                    "RM-DECOMPOSITION-TYPE",
                    "unsupported decomposition type or option",
                    1,
                ));
            }
            if ["lu", "chol", "triangular", "diagonal"].contains(&kind.as_str()) {
                if let Some(dims) = matrix.shape.known_dims() {
                    if dims.len() == 2
                        && dims[0].is_some()
                        && dims[1].is_some()
                        && dims[0] != dims[1]
                    {
                        diagnostics.push(argument_error(
                            "RM-DECOMPOSITION-SQUARE",
                            "selected decomposition type requires a square matrix",
                            0,
                        ));
                    }
                }
            }
        }
        for index in 1..request.arguments.len() {
            if request.literals.literal_string_at(index).as_deref() == Some("ranktolerance") {
                if request
                    .literals
                    .numeric_at(index + 1)
                    .is_some_and(|n| n < 0.0 || n.is_nan())
                {
                    diagnostics.push(argument_error(
                        "RM-DECOMPOSITION-TOLERANCE",
                        "RankTolerance must be nonnegative",
                        index + 1,
                    ));
                }
            }
        }
    } else {
        diagnostics.push(argument_error(
            "RM-DECOMPOSITION-ARITY",
            "expected a coefficient matrix",
            0,
        ));
    }
    finish_fixed(entry, request, object(), diagnostics)
}

pub(super) fn unary(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 1);
    receiver(request, 0, &mut diagnostics);
    // Reconstruct stable property facts; transpose and scale invalidate values.
    finish_fixed(entry, request, object(), diagnostics)
}

pub(super) fn scale(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 2);
    for (index, value) in request.arguments.iter().enumerate() {
        if !is_object(value) && value.shape.element_count().is_some_and(|n| n != 1) {
            diagnostics.push(argument_error(
                "RM-DECOMPOSITION-SCALE",
                "scale factor must be scalar",
                index,
            ));
        }
    }
    finish_fixed(entry, request, object(), diagnostics)
}

pub(super) fn right_divide(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = arity(request, 2);
    let result = if request.arguments.first().is_some_and(is_object) {
        object()
    } else {
        // Right solve precision, complexity and dimensions require the opaque
        // coefficient payload; input object shape is not coefficient shape.
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed(entry, request, result, diagnostics)
}

pub(super) fn left_divide(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 2);
    receiver(request, 0, &mut diagnostics);
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        diagnostics,
    )
}

pub(super) fn property(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 2);
    receiver(request, 0, &mut diagnostics);
    // LiteralContext cannot encode a substruct path, including subsequent
    // indexing. Do not infer the property from the shape of that path value.
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        }
    }
    fn matrix(rows: usize, cols: usize) -> ValueFact {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(rows), Some(cols)]),
            StorageFact::Dense,
        )
    }
    #[test]
    fn known_and_unknown_coefficients_return_host_value_objects() {
        for input in [
            matrix(0, 3),
            matrix(2, 2),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ] {
            let result = constructor(
                &request(vec![input]),
                &super::super::DECOMPOSITION_CATALOG_ENTRY,
            );
            assert!(is_object(&result.outputs[0]));
            assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
            assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        }
    }
    #[test]
    fn options_and_invalid_inputs_are_checked() {
        let mut req = request(vec![matrix(2, 3), ValueFact::scalar(ValueKindFact::String)]);
        req.literals = LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("chol".into()),
        ]);
        assert!(
            !constructor(&req, &super::super::DECOMPOSITION_CATALOG_ENTRY)
                .diagnostics
                .is_empty()
        );
        req.literals.literal_args[1] = LiteralValue::String("qr".into());
        assert!(
            constructor(&req, &super::super::DECOMPOSITION_CATALOG_ENTRY)
                .diagnostics
                .is_empty()
        );
        assert!(
            !constructor(&request(vec![]), &super::super::DECOMPOSITION_CATALOG_ENTRY)
                .diagnostics
                .is_empty()
        );
        assert!(!constructor(
            &request(vec![ValueFact::scalar(ValueKindFact::String)]),
            &super::super::DECOMPOSITION_CATALOG_ENTRY
        )
        .diagnostics
        .is_empty());
    }
    #[test]
    fn right_division_does_not_confuse_scaling_with_solving() {
        let entry = &super::super::DECOMPOSITION_MRDIVIDE_CATALOG_ENTRY;
        let scaled = right_divide(&request(vec![object(), matrix(1, 1)]), entry);
        assert!(is_object(&scaled.outputs[0]));
        let solved = right_divide(&request(vec![matrix(1, 2), object()]), entry);
        assert_eq!(solved.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(solved.outputs[0].residency, ResidencyFact::Unknown);
    }
    #[test]
    fn output_selection_retains_validation() {
        let mut req = request(vec![]);
        req.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
        assert!(
            !constructor(&req, &super::super::DECOMPOSITION_CATALOG_ENTRY)
                .diagnostics
                .is_empty()
        );
        req.arguments.push(matrix(1, 1));
        req.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(
            !constructor(&req, &super::super::DECOMPOSITION_CATALOG_ENTRY)
                .diagnostics
                .is_empty()
        );
    }
}
