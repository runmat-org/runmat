use crate::catalog::inference::{argument_error, literal_text};
use runmat_types::*;

#[derive(Clone, Copy, PartialEq)]
enum LabelDomain {
    Floating,
    Integer(NumericClass),
    Logical,
    Text,
}
fn domain(value: &ValueFact) -> Option<LabelDomain> {
    match &value.kind {
        ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real => {
            Some(if n.class.integer_class().is_some() {
                LabelDomain::Integer(n.class)
            } else {
                LabelDomain::Floating
            })
        }
        ValueKindFact::Logical => Some(LabelDomain::Logical),
        ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Cell(_) => {
            Some(LabelDomain::Text)
        }
        ValueKindFact::Object(object)
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| class.is(runmat_types::standard::CATEGORICAL)) =>
        {
            Some(LabelDomain::Text)
        }
        _ => None,
    }
}
fn label_count(value: &ValueFact) -> Option<usize> {
    if value.kind == ValueKindFact::Character {
        match &value.shape {
            ShapeFact::Scalar => Some(1),
            ShapeFact::Shaped { dims } => match dims.first() {
                Some(DimensionFact::Known(n)) => Some(*n),
                _ => None,
            },
            _ => None,
        }
    } else if matches!(
        value.kind,
        ValueKindFact::Object(_) | ValueKindFact::Unknown
    ) {
        None
    } else {
        value.shape.element_count()
    }
}
pub(super) fn validate(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    for (index, value) in request.arguments.iter().take(2).enumerate() {
        validate_labels(value, index, diagnostics);
    }
    if let [left, right, ..] = request.arguments.as_slice() {
        if matches!((label_count(left),label_count(right)),(Some(a),Some(b)) if a!=b) {
            diagnostics.push(argument_error(
                "RM-CATALOG-CONFUSIONMAT-LENGTH",
                "group and grouphat must contain the same number of labels",
                1,
            ));
        }
        if matches!((domain(left),domain(right)),(Some(a),Some(b)) if a!=b) {
            diagnostics.push(argument_error(
                "RM-CATALOG-CONFUSIONMAT-CLASS",
                "group and grouphat label classes must be compatible",
                1,
            ));
        }
    }
    for index in (2..request.arguments.len().saturating_sub(1)).step_by(2) {
        if let Some(name) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        {
            if super::canonical(&name) == "order" {
                let value = &request.arguments[index + 1];
                validate_labels(value, index + 1, diagnostics);
                if matches!((request.arguments.first().and_then(domain),domain(value)),(Some(a),Some(b)) if a!=b)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CONFUSIONMAT-ORDER-CLASS",
                        "Order label class must match group",
                        index + 1,
                    ));
                }
                if label_count(value) == Some(0) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CONFUSIONMAT-EMPTY-ORDER",
                        "Order must contain at least one label",
                        index + 1,
                    ));
                }
            }
        }
    }
}
fn validate_labels(value: &ValueFact, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let invalid_kind = match &value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Object(object) => object
            .runtime_class
            .as_ref()
            .is_some_and(|class| !class.is(runmat_types::standard::CATEGORICAL)),
        _ => domain(value).is_none(),
    };
    let invalid_shape = if value.kind == ValueKindFact::Character {
        value
            .shape
            .known_dims()
            .is_some_and(|dims| dims.iter().skip(2).any(|d| d.is_some_and(|d| d != 1)))
    } else if matches!(
        value.kind,
        ValueKindFact::Unknown | ValueKindFact::Object(_)
    ) {
        false
    } else {
        value
            .shape
            .known_dims()
            .is_some_and(|dims| dims.iter().filter(|d| d.is_some_and(|d| d > 1)).count() > 1)
    };
    if invalid_kind || invalid_shape || value.storage == StorageFact::Sparse {
        diagnostics.push(argument_error(
            "RM-CATALOG-CONFUSIONMAT-LABELS",
            "labels must be supported real vectors or two-dimensional character rows",
            index,
        ));
    }
}
