//! Conservative solver results; observed category counts are generally runtime values.
mod options;

use super::{MnrfitConstant, MnrfitOption};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, StructFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let len = request.arguments.len();
    if len < 2 || !len.saturating_sub(2).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MNRFIT-ARITY",
            "mnrfit requires X, Y, and optional name-value pairs",
            len,
        ));
    }
    if let Some(x) = request.arguments.first() {
        if x.storage == StorageFact::Sparse {
            diagnostics.push(argument_error(
                "RM-CATALOG-MNRFIT-X-STORAGE",
                "X does not accept sparse storage",
                0,
            ));
        }
        if !matches!(
            x.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-MNRFIT-X",
                "X must be a real numeric predictor matrix",
                0,
            ));
        }
        if x.shape.element_count() == Some(0)
            || first_extent(&x.shape) == Some(0)
            || second_extent(&x.shape) == Some(0)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-MNRFIT-X-EMPTY",
                "X must be nonempty",
                0,
            ));
        }
        if definitely_has_component_axis(&x.shape) {
            diagnostics.push(argument_error(
                "RM-CATALOG-MNRFIT-X-SHAPE",
                "X must be a 2-D predictor matrix",
                0,
            ));
        }
    }
    if let Some(y) = request.arguments.get(1) {
        if y.storage == StorageFact::Sparse {
            diagnostics.push(argument_error(
                "RM-CATALOG-MNRFIT-Y-STORAGE",
                "Y does not accept sparse storage",
                1,
            ));
        }
        let admitted = match &y.kind {
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Logical
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Cell(_)
            | ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_none_or(|class| class == &standard::CATEGORICAL.owned()),
            _ => false,
        };
        if !admitted {
            diagnostics.push(argument_error(
                "RM-CATALOG-MNRFIT-Y",
                "Y must be a numeric, logical, text, categorical, or cell-string response",
                1,
            ));
        }
        if request
            .arguments
            .first()
            .and_then(|x| first_extent(&x.shape))
            .is_some_and(|rows| definitely_invalid_response_shape(&y.kind, &y.shape, rows))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-MNRFIT-Y-SHAPE",
                "Y must have one response per row of X or valid grouped counts",
                1,
            ));
        }
    }
    for index in (2..len.saturating_sub(1)).step_by(2) {
        if request
            .arguments
            .get(index)
            .is_some_and(options::definitely_invalid_name)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-MNRFIT-OPTION",
                "mnrfit option names must be scalar text",
                index,
            ));
        }
        let Some(name) = request.literals.literal_string_at(index) else {
            continue;
        };
        let Some(identity) = MnrfitOption::from_public(&name) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-MNRFIT-OPTION",
                "unknown mnrfit option",
                index,
            ));
            continue;
        };
        if request.arguments.get(index + 1).is_some_and(|value| {
            options::definitely_invalid_value(identity, value)
                || (identity == MnrfitOption::Constant
                    && options::definitely_invalid_constant_literal(
                        value,
                        request.literals.literal_args.get(index + 1),
                    ))
        }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-MNRFIT-OPTION-VALUE",
                "mnrfit option value has an unsupported type or shape",
                index + 1,
            ));
        }
    }
    let b_shape = grouped_response_shape(request, options::constant_state(request))
        .unwrap_or_else(|| ShapeFact::from(vec![None, None]));
    let numerical = |shape| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Dense,
        )
    };
    let covariance_shape = coefficient_count(&b_shape)
        .map(|count| ShapeFact::from(vec![Some(count), Some(count)]))
        .unwrap_or_else(|| ShapeFact::from(vec![None, None]));
    let mut stat_fields = BTreeMap::new();
    for name in ["beta", "se", "t", "p"] {
        stat_fields.insert(name.to_string(), numerical(b_shape.clone()));
    }
    for name in ["covb", "coeffcorr"] {
        stat_fields.insert(name.to_string(), numerical(covariance_shape.clone()));
    }
    for name in ["dfe", "s"] {
        stat_fields.insert(
            name.to_string(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        );
    }
    stat_fields.insert(
        "estdisp".to_string(),
        ValueFact::scalar(ValueKindFact::Logical),
    );
    stat_fields.insert(
        "classNames".to_string(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    );
    let mut contract = CallContract::fixed(vec![
        numerical(b_shape),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        ValueFact::proven(
            ValueKindFact::Struct(StructFact::scalar(stat_fields, true)),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        ),
    ]);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !diagnostics.is_empty() {
        for output in &mut result.outputs {
            *output = ValueFact::unknown(DynamicReason::RuntimeValue);
        }
    }
    result.diagnostics.extend(diagnostics);
    result
}

fn definitely_invalid_response_shape(kind: &ValueKindFact, shape: &ShapeFact, rows: usize) -> bool {
    match kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) => {
            let grouped = first_two_extents(shape).is_some_and(|(y_rows, y_cols)| {
                y_rows == rows && y_cols >= 2 && !definitely_has_component_axis(shape)
            });
            !grouped && definitely_invalid_vector_shape(shape, rows)
        }
        ValueKindFact::Logical | ValueKindFact::String | ValueKindFact::Cell(_) => {
            definitely_invalid_vector_shape(shape, rows)
        }
        ValueKindFact::Character => first_extent(shape).is_some_and(|y_rows| y_rows != rows),
        _ => false,
    }
}

fn definitely_invalid_vector_shape(shape: &ShapeFact, rows: usize) -> bool {
    shape.element_count().is_some_and(|count| count != rows)
        || matches!(shape, ShapeFact::Shaped { dims } if dims.iter().filter(|dim| matches!(dim, DimensionFact::Known(extent) if *extent > 1)).count() > 1)
}

fn definitely_has_component_axis(shape: &ShapeFact) -> bool {
    matches!(shape, ShapeFact::Shaped { dims } if dims.iter().skip(2).any(|dim| matches!(dim, DimensionFact::Known(extent) if *extent != 1)))
}

fn first_extent(shape: &ShapeFact) -> Option<usize> {
    match shape {
        ShapeFact::Scalar => Some(1),
        ShapeFact::Shaped { dims } => dims.first().map(known_extent).unwrap_or(Some(1)),
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => None,
    }
}

fn second_extent(shape: &ShapeFact) -> Option<usize> {
    match shape {
        ShapeFact::Scalar => Some(1),
        ShapeFact::Shaped { dims } => dims.get(1).map(known_extent).unwrap_or(Some(1)),
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => None,
    }
}

fn first_two_extents(shape: &ShapeFact) -> Option<(usize, usize)> {
    match shape {
        ShapeFact::Scalar => Some((1, 1)),
        ShapeFact::Shaped { dims } => Some((
            dims.first().map(known_extent).unwrap_or(Some(1))?,
            dims.get(1).map(known_extent).unwrap_or(Some(1))?,
        )),
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => None,
    }
}

fn known_extent(extent: &DimensionFact) -> Option<usize> {
    match extent {
        DimensionFact::Known(value) => Some(*value),
        DimensionFact::Unknown | DimensionFact::Symbolic(_) => None,
    }
}

fn coefficient_count(shape: &ShapeFact) -> Option<usize> {
    let (rows, columns) = first_two_extents(shape)?;
    rows.checked_mul(columns)
}

fn grouped_response_shape(request: &CallRequest, constant: Option<bool>) -> Option<ShapeFact> {
    // A vector Y needs its runtime class census. A known N-by-K numeric matrix
    // fixes K grouped-count categories, provided Constant is admitted and known.
    let constant = constant?;
    let x = request.arguments.first()?;
    let y = request.arguments.get(1)?;
    if !matches!(
        y.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
    ) {
        return None;
    }
    let ShapeFact::Shaped { dims: x_dims } = &x.shape else {
        return None;
    };
    let ShapeFact::Shaped { dims: y_dims } = &y.shape else {
        return None;
    };
    if x_dims.len() < 2
        || y_dims.len() < 2
        || !x_dims
            .iter()
            .skip(2)
            .all(|dim| *dim == DimensionFact::Known(1))
        || !y_dims
            .iter()
            .skip(2)
            .all(|dim| *dim == DimensionFact::Known(1))
    {
        return None;
    }
    let (
        DimensionFact::Known(x_rows),
        DimensionFact::Known(x_cols),
        DimensionFact::Known(y_rows),
        DimensionFact::Known(y_cols),
    ) = (&x_dims[0], &x_dims[1], &y_dims[0], &y_dims[1])
    else {
        return None;
    };
    if x_rows != y_rows || *y_cols < 2 || *x_rows == 0 || *x_cols == 0 {
        return None;
    }
    Some(ShapeFact::from(vec![
        x_cols.checked_add(usize::from(constant)),
        Some(y_cols - 1),
    ]))
}
