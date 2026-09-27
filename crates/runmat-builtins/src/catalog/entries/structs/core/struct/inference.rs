use crate::catalog::inference::{argument_error, finish_fixed, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.as_slice() {
        [] => structure(BTreeMap::new(), true, ShapeFact::Scalar),
        [template] => match &template.kind {
            ValueKindFact::Struct(_) => template.clone(),
            ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Logical
                if template.shape.element_count() == Some(0)
                    && template.storage != StorageFact::Sparse =>
            {
                structure(
                    BTreeMap::new(),
                    true,
                    ShapeFact::from(vec![Some(0), Some(0)]),
                )
            }
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Logical
                if template.shape.element_count().is_none()
                    && template.storage != StorageFact::Sparse =>
            {
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRUCT-TEMPLATE",
                    "struct requires a struct or an empty real numeric/logical template",
                    0,
                ));
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            }
        },
        arguments if arguments.len() % 2 == 0 => pairs(request, &mut diagnostics),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRUCT-PAIRS",
                "struct requires complete field/value pairs",
                request.arguments.len() - 1,
            ));
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn pairs(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> ValueFact {
    let mut fields = BTreeMap::new();
    let mut complete = true;
    let mut shape: Option<ShapeFact> = None;
    let mut unknown_container = false;
    for index in (0..request.arguments.len()).step_by(2) {
        let value = &request.arguments[index + 1];
        let field = match &value.kind {
            ValueKindFact::Cell(cell) => {
                if let Some(prior) = &shape {
                    if let (Some(a), Some(b)) = (prior.known_dims(), value.shape.known_dims()) {
                        if a.len() != b.len()
                            || a.iter()
                                .zip(&b)
                                .any(|(a, b)| matches!((a,b),(Some(a),Some(b)) if a != b))
                        {
                            diagnostics.push(argument_error(
                                "RM-CATALOG-STRUCT-CELLS",
                                "cell-valued fields must have matching shapes",
                                index + 1,
                            ));
                        }
                    }
                } else {
                    shape = Some(value.shape.clone());
                }
                (*cell.element).clone()
            }
            ValueKindFact::Unknown => {
                unknown_container = true;
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
            _ => value.clone(),
        };
        if let Some(name) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        {
            if fields.insert(name, field).is_some() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRUCT-DUPLICATE",
                    "struct field names must be unique",
                    index,
                ));
            }
        } else {
            complete = false;
        }
    }
    let shape = if unknown_container {
        ShapeFact::Unknown
    } else {
        shape.unwrap_or(ShapeFact::Scalar)
    };
    let mut output = structure(fields, complete, shape);
    // Position-specific field facts retain nested device/foreign/callable facts
    // without inspecting or materializing any runtime payload.
    if complete {
        if let Some(elements) = array_elements(request, output.shape.element_count()) {
            if let ValueKindFact::Struct(fact) = &mut output.kind {
                fact.elements = elements;
                fact.elements_complete = true;
            }
        }
    }
    output
}

fn array_elements(
    request: &CallRequest,
    count: Option<usize>,
) -> Option<Vec<BTreeMap<String, ValueFact>>> {
    let count = count?;
    // Allocate facts only when a complete cell already supplies each position.
    let cells: Vec<_> = request
        .arguments
        .iter()
        .skip(1)
        .step_by(2)
        .filter_map(|value| match &value.kind {
            ValueKindFact::Cell(cell) => Some(cell),
            _ => None,
        })
        .collect();
    if cells.is_empty() && count != 1 {
        return None;
    }
    if cells
        .iter()
        .any(|cell| !cell.elements_complete || cell.elements.len() != count)
    {
        return None;
    }
    let mut elements = vec![BTreeMap::new(); count];
    for (position, element) in elements.iter_mut().enumerate() {
        for index in (0..request.arguments.len()).step_by(2) {
            let name = request
                .literals
                .literal_args
                .get(index)
                .and_then(literal_text)?;
            let value = &request.arguments[index + 1];
            let field = match &value.kind {
                ValueKindFact::Cell(cell) => cell.elements[position].clone(),
                ValueKindFact::Unknown => return None,
                _ => value.clone(),
            };
            element.insert(name, field);
        }
    }
    Some(elements)
}

fn structure(fields: BTreeMap<String, ValueFact>, complete: bool, shape: ShapeFact) -> ValueFact {
    let count = shape.element_count();
    let scalar = shape.element_count() == Some(1);
    let fact = if scalar {
        StructFact::scalar(fields, complete)
    } else {
        StructFact::array(
            fields,
            complete,
            Vec::new(),
            shape.element_count() == Some(0),
        )
    };
    ValueFact::proven(
        ValueKindFact::Struct(fact),
        shape,
        if scalar {
            StorageFact::Scalar
        } else if count.is_none() {
            StorageFact::Unknown
        } else {
            StorageFact::Dense
        },
    )
}
