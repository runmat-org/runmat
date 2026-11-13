use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, NumericClass, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

type MatrixDims = Option<(Option<usize>, Option<usize>)>;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let arguments = &request.arguments;
    let (a, b, b_index) = match arguments.as_slice() {
        [model]
            if matches!(
                model.kind,
                ValueKindFact::Object(_) | ValueKindFact::Unknown
            ) =>
        {
            if let ValueKindFact::Object(object) = &model.kind {
                let valid_model = match object.runtime_class.as_ref() {
                    Some(class) => class == &standard::STATE_SPACE.owned(),
                    None => object.class.is_none(),
                };
                if !valid_model
                    || model.shape.element_count().is_some_and(|count| count != 1)
                    || matches!(model.residency, ResidencyFact::Remote { .. })
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CTRB-MODEL",
                        "sys must be one ordinary state-space ss model",
                        0,
                    ));
                }
                if object.properties_complete
                    && (!object.properties.contains_key("A")
                        || !object.properties.contains_key("B"))
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CTRB-MODEL",
                        "state-space model must contain A and B",
                        0,
                    ));
                }
                (object.properties.get("A"), object.properties.get("B"), 0)
            } else {
                (None, None, 0)
            }
        }
        [a, b] => (Some(a), Some(b), 1),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-CTRB-ARITY",
                "ctrb expects (A,B) or (sys)",
                arguments.len().saturating_sub(1),
            ));
            (None, None, 1)
        }
    };
    let a_dims = a.and_then(|value| matrix_dims(value, 0, &mut diagnostics));
    let b_dims = b.and_then(|value| matrix_dims(value, b_index, &mut diagnostics));
    let n = a_dims.flatten().and_then(|(rows, columns)| {
        if rows
            .zip(columns)
            .is_some_and(|(rows, columns)| rows != columns)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-CTRB-DIMENSIONS",
                "A must be square",
                0,
            ));
        }
        rows
    });
    let m = b_dims.flatten().and_then(|(_, columns)| columns);
    if n == Some(0) || m == Some(0) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CTRB-DIMENSIONS",
            "A must have at least one state and B at least one input",
            b_index,
        ));
    }
    if let (Some(n), Some((Some(rows), _))) = (n, b_dims.flatten()) {
        if rows != n {
            diagnostics.push(argument_error(
                "RM-CATALOG-CTRB-DIMENSIONS",
                "B rows must match A",
                b_index,
            ));
        }
    }
    if let (Some(a), Some(b)) = (
        a.and_then(ValueFact::numeric),
        b.and_then(ValueFact::numeric),
    ) {
        if a != b {
            diagnostics.push(argument_error(
                "RM-CATALOG-CTRB-CLASS",
                "A and B must have matching floating precision and real/complex domain",
                b_index,
            ));
        }
    }
    let shape = ShapeFact::from(vec![n, n.zip(m).and_then(|(n, m)| n.checked_mul(m))]);
    let output = if let Some(numeric) = b
        .and_then(ValueFact::numeric)
        .filter(|numeric| matches!(numeric.class, NumericClass::Single | NumericClass::Double))
    {
        ValueFact::proven(ValueKindFact::Numeric(numeric), shape, StorageFact::Dense)
    } else {
        let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
        output.shape = shape;
        output.storage = StorageFact::Dense;
        output
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn matrix_dims(
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<MatrixDims> {
    if matches!(value.residency, ResidencyFact::Remote { .. })
        || !matches!(value.storage, StorageFact::Dense | StorageFact::Unknown)
        || (!matches!(value.kind, ValueKindFact::Unknown)
            && !value.numeric().is_some_and(|numeric| {
                matches!(numeric.class, NumericClass::Single | NumericClass::Double)
            }))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-CTRB-INPUT",
            "matrix must have dense real or complex single/double numeric storage",
            index,
        ));
        return None;
    }
    if value.shape.known_dims().is_some_and(|dims| {
        dims.iter()
            .skip(2)
            .any(|extent| extent.is_some_and(|extent| extent != 1))
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CTRB-DIMENSIONS",
            "matrix dimensions after the second must be singleton",
            index,
        ));
        return None;
    }
    Some(value.shape.known_dims().map(|dims| {
        (
            dims.first().copied().unwrap_or(Some(1)),
            dims.get(1).copied().unwrap_or(Some(1)),
        )
    }))
}
