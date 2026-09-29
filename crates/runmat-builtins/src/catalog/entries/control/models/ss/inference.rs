use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, CellFact, InferenceDiagnostic, LiteralValue,
    NumericClass, NumericDomain, NumericFact, ObjectFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
    if args.len() < 4 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SS-ARITY",
            "ss requires A, B, C, and D",
            args.len(),
        ));
    }
    let matrices: Vec<_> = args
        .iter()
        .take(4)
        .enumerate()
        .map(|(index, value)| {
            let shape = matrix_shape(value);
            if shape.is_none() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SS-MATRIX",
                    "state-space matrices must be finite real numeric scalars or 2-D arrays",
                    index,
                ));
            }
            shape.flatten()
        })
        .collect();
    if matrices.len() == 4 {
        validate_dimensions(&matrices, &mut diagnostics);
    }
    validate_options(request, &mut diagnostics);

    let mut properties = BTreeMap::new();
    for (name, shape) in ["A", "B", "C", "D"].into_iter().zip(matrices) {
        properties.insert(name.to_string(), real_matrix(shape));
    }
    // Even with unknown inputs the constructed object has the complete property schema.
    for name in ["A", "B", "C", "D"] {
        properties
            .entry(name.to_string())
            .or_insert_with(|| real_matrix(None));
    }
    properties.insert("Ts".into(), real_scalar());
    let states = property_dim(&properties, "A", 0);
    let inputs = property_dim(&properties, "B", 1);
    let outputs = property_dim(&properties, "C", 0);
    for (name, length) in [("InputDelay", inputs), ("OutputDelay", outputs)] {
        properties.insert(name.into(), real_matrix(Some((length, Some(1)))));
    }
    for (name, length) in [
        ("StateName", states),
        ("InputName", inputs),
        ("OutputName", outputs),
    ] {
        properties.insert(name.into(), name_cells(length));
    }
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::STATE_SPACE.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, output, diagnostics)
}

type MatrixDims = Option<(Option<usize>, Option<usize>)>;

fn matrix_shape(value: &ValueFact) -> Option<MatrixDims> {
    if value.storage == StorageFact::Sparse {
        return None;
    }
    match &value.kind {
        ValueKindFact::Unknown => {}
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {}
        _ => return None,
    }
    let Some(dims) = value.shape.known_dims() else {
        return Some(None);
    };
    match dims.as_slice() {
        [] => Some(Some((Some(1), Some(1)))),
        [columns] => Some(Some((Some(1), *columns))),
        [rows, columns] => Some(Some((*rows, *columns))),
        _ => None,
    }
}

fn validate_dimensions(shapes: &[MatrixDims], diagnostics: &mut Vec<InferenceDiagnostic>) {
    let [Some(a), Some(b), Some(c), Some(d)] = shapes else {
        return;
    };
    for (left, right, label) in [
        (a.0, a.1, "A must be square"),
        (b.0, a.0, "B rows must match A"),
        (c.1, a.0, "C columns must match A"),
        (d.0, c.0, "D rows must match C"),
        (d.1, b.1, "D columns must match B"),
    ] {
        if left.zip(right).is_some_and(|(left, right)| left != right) {
            diagnostics.push(argument_error("RM-CATALOG-SS-DIMENSIONS", label, 0));
        }
    }
}

fn validate_options(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let rest = request.arguments.get(4..).unwrap_or(&[]);
    if rest.is_empty() {
        return;
    }
    if rest.len() == 1 {
        validate_sample_time(
            &rest[0],
            request.literals.literal_args.get(4),
            4,
            diagnostics,
        );
        return;
    }
    if !rest.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SS-OPTIONS",
            "options require a scalar sample time or name/value pairs",
            4,
        ));
        return;
    }
    for (pair_index, pair) in rest.chunks_exact(2).enumerate() {
        let index = 4 + pair_index * 2;
        if !text_scalar_or_unknown(&pair[0]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SS-OPTION",
                "option names must be text",
                index,
            ));
        }
        if let Some(name) = request.literals.literal_string_at(index) {
            if !matches!(name.trim(), "ts" | "sampletime") {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SS-OPTION",
                    "unsupported sample-time option",
                    index,
                ));
            }
        }
        validate_sample_time(
            &pair[1],
            request.literals.literal_args.get(index + 1),
            index + 1,
            diagnostics,
        );
    }
}

fn text_scalar_or_unknown(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => possible_text_shape(value),
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Character => possible_text_shape(value),
        _ => false,
    }
}

fn possible_text_shape(value: &ValueFact) -> bool {
    value.shape.known_dims().is_none_or(|dims| {
        dims.len() <= 2 && dims.first().copied().flatten().is_none_or(|rows| rows == 1)
    })
}

fn validate_sample_time(
    value: &ValueFact,
    literal: Option<&LiteralValue>,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !matches!(value.kind, ValueKindFact::Unknown)
        && (!value
            .numeric()
            .is_some_and(|n| n.domain == NumericDomain::Real)
            || value.shape.element_count().is_some_and(|count| count != 1))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SS-SAMPLE-TIME",
            "Ts must be a real numeric scalar",
            index,
        ));
    }
    let known = match literal {
        Some(LiteralValue::Number(value)) => Some(*value),
        Some(LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. }) => {
            text.parse::<f64>().ok()
        }
        _ => None,
    };
    if known.is_some_and(|value| !value.is_finite() || value < 0.0) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SS-SAMPLE-TIME",
            "Ts must be finite and nonnegative",
            index,
        ));
    }
}

fn real_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn real_matrix(shape: MatrixDims) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![shape.and_then(|s| s.0), shape.and_then(|s| s.1)]),
        StorageFact::Dense,
    )
}

fn property_dim(
    properties: &BTreeMap<String, ValueFact>,
    name: &str,
    axis: usize,
) -> Option<usize> {
    properties
        .get(name)?
        .shape
        .known_dims()?
        .get(axis)
        .copied()
        .flatten()
}

fn name_cells(rows: Option<usize>) -> ValueFact {
    let element = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(0)]),
        StorageFact::Dense,
    );
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: vec![],
            elements_complete: false,
        }),
        ShapeFact::from(vec![rows, Some(1)]),
        StorageFact::Dense,
    )
}
