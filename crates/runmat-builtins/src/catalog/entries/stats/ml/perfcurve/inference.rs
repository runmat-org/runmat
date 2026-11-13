//! Conservative, option-aware public facts for performance curves.

use super::{
    PerfcurveBoolean, PerfcurveCriterion, PerfcurveOptionName, PerfcurvePriorKeyword,
    PerfcurveProcessNaN,
};
use crate::catalog::inference::{argument_error, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn diagnostic(code: &'static str, message: &'static str, index: usize) -> InferenceDiagnostic {
    argument_error(code, message, index)
}

fn real_numeric(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
    )
}

fn known_vector_invalid(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Shaped { dims } => {
            dims.iter()
                .filter(|dim| matches!(dim, DimensionFact::Known(n) if *n > 1))
                .count()
                > 1
        }
        _ => false,
    }
}

fn vector_geometry_proven(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Scalar => true,
        ShapeFact::Shaped { dims } => {
            dims.iter()
                .filter(|dim| !matches!(dim, DimensionFact::Known(1)))
                .count()
                <= 1
                && dims
                    .iter()
                    .all(|dim| matches!(dim, DimensionFact::Known(_)))
        }
        _ => false,
    }
}

fn label_count(value: &ValueFact) -> Option<usize> {
    match &value.kind {
        ValueKindFact::Character => match &value.shape {
            ShapeFact::Scalar => Some(1),
            ShapeFact::Shaped { dims } => match dims.first() {
                Some(DimensionFact::Known(n)) => Some(*n),
                _ => None,
            },
            _ => None,
        },
        ValueKindFact::Object(_) | ValueKindFact::Unknown => None,
        _ => value.shape.element_count(),
    }
}

fn label_kind_known_invalid(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Numeric(n) => n.domain != NumericDomain::Real,
        ValueKindFact::Logical
        | ValueKindFact::String
        | ValueKindFact::Character
        | ValueKindFact::Cell(_)
        | ValueKindFact::Unknown => false,
        ValueKindFact::Object(object) => object
            .runtime_class
            .as_ref()
            .is_some_and(|class| !class.is(runmat_types::standard::CATEGORICAL)),
        _ => true,
    }
}

fn use_nearest_literal_valid(literal: &LiteralValue) -> Option<bool> {
    match literal {
        LiteralValue::Unknown => None,
        LiteralValue::Bool(_) => Some(true),
        LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. } => Some(
            LiteralContext::numeric_from_literal(literal)
                .is_some_and(|value| value == 0.0 || value == 1.0),
        ),
        _ => Some(
            literal_text(literal)
                .is_some_and(|text| PerfcurveBoolean::from_public(&text).is_some()),
        ),
    }
}

fn validate_required(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> bool {
    let mut dynamic = false;
    if request.arguments.len() < 3 {
        diagnostics.push(diagnostic(
            "RM-CATALOG-PERFCURVE-ARITY",
            "perfcurve requires labels, scores, and posclass",
            request.arguments.len(),
        ));
        return true;
    }
    let labels = &request.arguments[0];
    let scores = &request.arguments[1];
    let positive = &request.arguments[2];
    for (index, value) in [(0, labels), (2, positive)] {
        if label_kind_known_invalid(value) {
            diagnostics.push(diagnostic(
                "RM-CATALOG-PERFCURVE-LABEL",
                "labels and posclass must have an admitted label class",
                index,
            ));
        }
        if matches!(
            value.kind,
            ValueKindFact::Unknown | ValueKindFact::Object(_) | ValueKindFact::Cell(_)
        ) {
            dynamic = true;
        }
        if index == 0
            && value.kind != ValueKindFact::Character
            && known_vector_invalid(&value.shape)
        {
            diagnostics.push(diagnostic(
                "RM-CATALOG-PERFCURVE-LABEL-SHAPE",
                "labels must be a vector",
                index,
            ));
        }
        if index == 0
            && value.kind != ValueKindFact::Character
            && !vector_geometry_proven(&value.shape)
        {
            dynamic = true;
        }
    }
    if (positive.kind == ValueKindFact::Character
        && label_count(positive).is_some_and(|rows| rows != 1))
        || (positive.kind != ValueKindFact::Character
            && positive.shape.element_count().is_some_and(|n| n != 1))
    {
        diagnostics.push(diagnostic(
            "RM-CATALOG-PERFCURVE-POSCLASS",
            "posclass must be one label",
            2,
        ));
    }
    if !real_numeric(scores) {
        if scores.kind == ValueKindFact::Unknown {
            dynamic = true;
        } else {
            diagnostics.push(diagnostic(
                "RM-CATALOG-PERFCURVE-SCORES",
                "scores must be a real numeric vector",
                1,
            ));
        }
    }
    if known_vector_invalid(&scores.shape) {
        diagnostics.push(diagnostic(
            "RM-CATALOG-PERFCURVE-SCORE-SHAPE",
            "scores must be a vector",
            1,
        ));
    }
    if !vector_geometry_proven(&scores.shape) {
        dynamic = true;
    }
    if scores.shape.element_count() == Some(0) {
        diagnostics.push(diagnostic(
            "RM-CATALOG-PERFCURVE-EMPTY",
            "scores must contain observations",
            1,
        ));
    }
    if label_count(labels)
        .zip(scores.shape.element_count())
        .is_some_and(|(a, b)| a != b)
    {
        diagnostics.push(diagnostic(
            "RM-CATALOG-PERFCURVE-LENGTH",
            "labels and scores must have the same number of observations",
            1,
        ));
    }
    dynamic
}

fn validate_options(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> bool {
    let mut dynamic = false;
    let rest = request.arguments.len().saturating_sub(3);
    if !rest.is_multiple_of(2) {
        diagnostics.push(diagnostic(
            "RM-CATALOG-PERFCURVE-PAIRS",
            "options must be name-value pairs",
            request.arguments.len(),
        ));
    }
    let mut has_tvals = false;
    let mut has_xvals = false;
    for index in (3..request.arguments.len().saturating_sub(1)).step_by(2) {
        let name_fact = &request.arguments[index];
        let Some(spelling) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        else {
            if !matches!(
                name_fact.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            ) {
                diagnostics.push(diagnostic(
                    "RM-CATALOG-PERFCURVE-OPTION-NAME",
                    "option names must be text",
                    index,
                ));
            }
            dynamic = true;
            continue;
        };
        let Some(option) = PerfcurveOptionName::from_public(&spelling) else {
            diagnostics.push(diagnostic(
                "RM-CATALOG-PERFCURVE-OPTION",
                "unsupported perfcurve option",
                index,
            ));
            continue;
        };
        if option == PerfcurveOptionName::UnsupportedBootstrap {
            diagnostics.push(diagnostic(
                "RM-CATALOG-PERFCURVE-UNSUPPORTED",
                "bootstrap and confidence-bound modes are unsupported",
                index,
            ));
            continue;
        }
        let value = &request.arguments[index + 1];
        let literal = request.literals.literal_args.get(index + 1);
        match option {
            PerfcurveOptionName::XCrit | PerfcurveOptionName::YCrit => {
                if let Some(text) = literal.and_then(literal_text) {
                    if PerfcurveCriterion::from_public(&text).is_none() {
                        diagnostics.push(diagnostic(
                            "RM-CATALOG-PERFCURVE-CRITERION",
                            "unsupported criterion",
                            index + 1,
                        ));
                    }
                } else if literal.is_some_and(|literal| !matches!(literal, LiteralValue::Unknown)) {
                    diagnostics.push(diagnostic(
                        "RM-CATALOG-PERFCURVE-CRITERION",
                        "criterion must be an admitted text scalar",
                        index + 1,
                    ));
                } else {
                    dynamic = true;
                }
            }
            PerfcurveOptionName::ProcessNaN => {
                if let Some(text) = literal.and_then(literal_text) {
                    if PerfcurveProcessNaN::from_public(&text).is_none() {
                        diagnostics.push(diagnostic(
                            "RM-CATALOG-PERFCURVE-PROCESSNAN",
                            "unsupported ProcessNaN value",
                            index + 1,
                        ));
                    }
                } else if literal.is_some_and(|literal| !matches!(literal, LiteralValue::Unknown)) {
                    diagnostics.push(diagnostic(
                        "RM-CATALOG-PERFCURVE-PROCESSNAN",
                        "ProcessNaN must be an admitted text scalar",
                        index + 1,
                    ));
                } else {
                    dynamic = true;
                }
            }
            PerfcurveOptionName::Prior => {
                if let Some(text) = literal.and_then(literal_text) {
                    if PerfcurvePriorKeyword::from_public(&text).is_none() {
                        diagnostics.push(diagnostic(
                            "RM-CATALOG-PERFCURVE-PRIOR",
                            "unsupported Prior value",
                            index + 1,
                        ));
                    }
                } else if real_numeric(value) {
                    if value.shape.element_count().is_some_and(|n| n != 2)
                        || known_vector_invalid(&value.shape)
                    {
                        diagnostics.push(diagnostic(
                            "RM-CATALOG-PERFCURVE-PRIOR-SHAPE",
                            "numeric Prior needs two values",
                            index + 1,
                        ));
                    }
                } else {
                    dynamic = true;
                }
            }
            PerfcurveOptionName::Cost => {
                let known_bad_shape = match &value.shape {
                    ShapeFact::Scalar
                    | ShapeFact::Ranked { rank: 1 }
                    | ShapeFact::Ranked { rank: 3.. } => true,
                    ShapeFact::Shaped { dims } => {
                        dims.len() != 2
                            || dims
                                .iter()
                                .any(|dim| matches!(dim, DimensionFact::Known(n) if *n != 2))
                    }
                    _ => false,
                };
                if known_bad_shape || !real_numeric(value) && value.kind != ValueKindFact::Unknown {
                    diagnostics.push(diagnostic(
                        "RM-CATALOG-PERFCURVE-COST",
                        "Cost must be a 2-by-2 real numeric matrix",
                        index + 1,
                    ));
                } else if value.kind == ValueKindFact::Unknown
                    || !matches!(&value.shape, ShapeFact::Shaped { dims } if dims.as_slice() == [DimensionFact::Known(2), DimensionFact::Known(2)])
                {
                    dynamic = true;
                }
            }
            PerfcurveOptionName::Weights
            | PerfcurveOptionName::TVals
            | PerfcurveOptionName::XVals => {
                if !real_numeric(value) {
                    if value.kind == ValueKindFact::Unknown {
                        dynamic = true;
                    } else {
                        diagnostics.push(diagnostic(
                            "RM-CATALOG-PERFCURVE-NUMERIC-OPTION",
                            "option requires real numeric values",
                            index + 1,
                        ));
                    }
                }
                if known_vector_invalid(&value.shape) {
                    diagnostics.push(diagnostic(
                        "RM-CATALOG-PERFCURVE-OPTION-SHAPE",
                        "numeric option must be a vector",
                        index + 1,
                    ));
                }
                if !vector_geometry_proven(&value.shape) {
                    dynamic = true;
                }
                if option == PerfcurveOptionName::Weights
                    && label_count(&request.arguments[0])
                        .zip(value.shape.element_count())
                        .is_some_and(|(a, b)| a != b)
                {
                    diagnostics.push(diagnostic(
                        "RM-CATALOG-PERFCURVE-WEIGHTS",
                        "Weights must match observations",
                        index + 1,
                    ));
                }
                has_tvals |= option == PerfcurveOptionName::TVals;
                has_xvals |= option == PerfcurveOptionName::XVals;
            }
            PerfcurveOptionName::UseNearest => {
                if !matches!(
                    value.kind,
                    ValueKindFact::Logical
                        | ValueKindFact::Numeric(_)
                        | ValueKindFact::String
                        | ValueKindFact::Character
                        | ValueKindFact::Unknown
                ) || value.shape.element_count().is_some_and(|n| n != 1)
                    || literal.and_then(use_nearest_literal_valid) == Some(false)
                {
                    diagnostics.push(diagnostic(
                        "RM-CATALOG-PERFCURVE-NEAREST",
                        "UseNearest must be scalar logical",
                        index + 1,
                    ));
                } else if value.kind == ValueKindFact::Unknown
                    || literal.and_then(use_nearest_literal_valid).is_none()
                {
                    dynamic = true;
                }
            }
            PerfcurveOptionName::NegClass => {
                if label_kind_known_invalid(value) {
                    diagnostics.push(diagnostic(
                        "RM-CATALOG-PERFCURVE-NEGCLASS",
                        "NegClass must contain labels",
                        index + 1,
                    ));
                }
                if value.kind == ValueKindFact::Unknown {
                    dynamic = true;
                }
            }
            PerfcurveOptionName::UnsupportedBootstrap => unreachable!(),
        }
    }
    if has_tvals && has_xvals {
        diagnostics.push(diagnostic(
            "RM-CATALOG-PERFCURVE-THRESHOLD-CONFLICT",
            "TVals and XVals cannot be combined",
            3,
        ));
    }
    dynamic
}

fn numeric(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let dynamic =
        validate_required(request, &mut diagnostics) | validate_options(request, &mut diagnostics);
    let invalid = !diagnostics.is_empty();
    let curve = ShapeFact::from(vec![None, Some(1)]);
    let mut values = vec![
        numeric(curve.clone()),
        numeric(curve.clone()),
        numeric(curve),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        numeric(ShapeFact::from(vec![Some(1), Some(2)])),
        numeric(ShapeFact::from(vec![None, None])),
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                elements: Vec::new(),
                elements_complete: false,
            }),
            ShapeFact::from(vec![None, Some(1)]),
            StorageFact::Dense,
        ),
    ];
    if invalid || dynamic {
        values
            .iter_mut()
            .for_each(|value| *value = ValueFact::unknown(DynamicReason::RuntimeValue));
    }
    let mut contract = CallContract::fixed(values);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
