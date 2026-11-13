use super::{DistanceControlKind, DistanceMetricKind, DistanceSelectionKind};
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy)]
enum SelectionFact {
    None,
    Selected { count: Option<usize> },
    Unknown,
}

fn matrix_shape(
    shape: &ShapeFact,
    argument: usize,
    errors: &mut Vec<InferenceDiagnostic>,
) -> [Option<usize>; 2] {
    match shape {
        ShapeFact::Scalar => [Some(1), Some(1)],
        ShapeFact::Shaped { dims } if dims.len() == 2 => [
            match dims[0] {
                DimensionFact::Known(value) => Some(value),
                _ => None,
            },
            match dims[1] {
                DimensionFact::Known(value) => Some(value),
                _ => None,
            },
        ],
        ShapeFact::Shaped { .. } => {
            errors.push(argument_error(
                "RM-CATALOG-PAIRWISE-RANK",
                "distance observations must be a vector or two-dimensional matrix",
                argument,
            ));
            [None, None]
        }
        ShapeFact::Ranked { rank: 2 } | ShapeFact::Unknown => [None, None],
        ShapeFact::Ranked { .. } => [None, None],
    }
}

fn check_data(
    value: Option<&ValueFact>,
    index: usize,
    errors: &mut Vec<InferenceDiagnostic>,
) -> [Option<usize>; 2] {
    let Some(value) = value else {
        return [None, None];
    };
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Unknown
    ) {
        errors.push(argument_error(
            "RM-CATALOG-PAIRWISE-CLASS",
            "distance observations must be real numeric arrays",
            index,
        ));
    }
    matrix_shape(&value.shape, index, errors)
}

fn double(shape: ShapeFact) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    if matches!(fact.shape, ShapeFact::Unknown) {
        fact.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    fact
}

pub(super) fn infer_pdist(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-PDIST-ARITY",
            "pdist requires X",
            0,
        ));
    }
    let [rows, _] = check_data(request.arguments.first(), 0, &mut diagnostics);
    validate_metric(request, 1, false, &mut diagnostics);
    let length = rows.and_then(super::super::distance_geometry::condensed_length);
    let output = double(ShapeFact::from(vec![Some(1), length]));
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn infer_pdist2(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PDIST2-ARITY",
            "pdist2 requires X and Y",
            request.arguments.len(),
        ));
    }
    let [x_rows, x_columns] = check_data(request.arguments.first(), 0, &mut diagnostics);
    let [y_rows, y_columns] = check_data(request.arguments.get(1), 1, &mut diagnostics);
    if x_columns.is_some() && y_columns.is_some() && x_columns != y_columns {
        diagnostics.push(argument_error(
            "RM-CATALOG-PDIST2-COLUMNS",
            "X and Y must have the same number of columns",
            1,
        ));
    }
    validate_metric(request, 2, true, &mut diagnostics);
    let selection = selection_fact(request, &mut diagnostics);
    let shape = match selection {
        SelectionFact::None => ShapeFact::from(vec![x_rows, y_rows]),
        SelectionFact::Selected { count } => ShapeFact::from(vec![
            count.and_then(|count| x_rows.map(|rows| rows.min(count))),
            y_rows,
        ]),
        SelectionFact::Unknown => ShapeFact::from(vec![None, y_rows]),
    };
    let first = double(shape.clone());
    let mut outputs = vec![first];
    if !matches!(selection, SelectionFact::None) {
        outputs.push(double(shape));
    } else if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-PDIST2-OUTPUTS",
            "second output requires Smallest or Largest",
            0,
        ));
    }
    let mut contract = CallContract::fixed(outputs);
    if !matches!(selection, SelectionFact::None) {
        // The runtime pads requested output slots beyond [D,I] with zero double scalars.
        contract.maximum_outputs = None;
        contract.variadic_output = Some(Box::new(ValueFact::scalar(ValueKindFact::Numeric(
            NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            },
        ))));
    }
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn selection_fact(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> SelectionFact {
    let mut selected = None;
    let mut unknown = false;
    let mut index = 2;
    while index < request.arguments.len() {
        let literal = request.literals.literal_args.get(index);
        let keyword = literal.and_then(|literal| match literal {
            LiteralValue::Character(text)
            | LiteralValue::String(text)
            | LiteralValue::Keyword(text) => Some(text.as_str()),
            _ => None,
        });
        if keyword
            .and_then(DistanceSelectionKind::from_keyword)
            .is_some()
        {
            if selected.is_some() || index + 1 >= request.arguments.len() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PDIST2-SELECTION",
                    "Smallest or Largest must be one name-value pair",
                    index,
                ));
                return SelectionFact::Unknown;
            }
            let count = match request.literals.literal_args.get(index + 1) {
                Some(
                    literal @ (LiteralValue::Number(_)
                    | LiteralValue::Real { .. }
                    | LiteralValue::Integer { .. }),
                ) => {
                    let parsed = positive_count(literal);
                    if parsed.is_none() {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-PDIST2-SELECTION-COUNT",
                            "Smallest or Largest count must be a positive exact integer",
                            index + 1,
                        ));
                    }
                    parsed
                }
                Some(_) | None => None,
            };
            selected = Some(SelectionFact::Selected { count });
            index += 2;
        } else {
            if keyword.is_none()
                && request.arguments.get(index).is_some_and(|fact| {
                    matches!(
                        fact.kind,
                        ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown
                    )
                })
            {
                unknown = true;
            }
            index += 1;
        }
    }
    if unknown {
        SelectionFact::Unknown
    } else {
        selected.unwrap_or(SelectionFact::None)
    }
}

fn literal_text(literal: &LiteralValue) -> Option<&str> {
    match literal {
        LiteralValue::Character(text)
        | LiteralValue::String(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

fn validate_metric(
    request: &CallRequest,
    start: usize,
    allow_selection: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let mut index = start;
    while index < request.arguments.len() {
        let Some(text) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        else {
            return;
        };
        if DistanceControlKind::from_keyword(text).is_some()
            || (allow_selection && DistanceSelectionKind::from_keyword(text).is_some())
        {
            index += 2;
            continue;
        }
        if DistanceMetricKind::from_keyword(text).is_none() {
            diagnostics.push(argument_error(
                "RM-CATALOG-PAIRWISE-METRIC",
                "unsupported named distance metric",
                index,
            ));
        }
        return;
    }
}

fn positive_count(literal: &LiteralValue) -> Option<usize> {
    match literal {
        LiteralValue::Integer { text, .. } => text.parse::<usize>().ok().filter(|value| *value > 0),
        LiteralValue::Number(value)
            if value.is_finite()
                && *value >= 1.0
                && value.fract() == 0.0
                && *value < usize::MAX as f64 =>
        {
            Some(*value as usize)
        }
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .and_then(|value| positive_count(&LiteralValue::Number(value))),
        _ => None,
    }
}
