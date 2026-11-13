use super::shape::{reduced_shape, tie_adjustment_shape, ReductionAxis};
use super::OrderStatisticOption;
use super::QuantileRequestUnit;
use super::TiedRankMode;
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn diagnostic(code: &'static str, message: &'static str, index: usize) -> InferenceDiagnostic {
    argument_error(code, message, index)
}

fn real_numeric_or_unknown(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    )
}

fn double_output(shape: ShapeFact) -> ValueFact {
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    if matches!(fact.shape, ShapeFact::Unknown | ShapeFact::Ranked { .. }) {
        fact.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    fact
}

fn finish(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    outputs: Vec<ValueFact>,
    diagnostics: Vec<InferenceDiagnostic>,
    padded: bool,
) -> CallInference {
    let mut contract = CallContract::fixed(outputs);
    if padded {
        contract.maximum_outputs = None;
        contract.variadic_output = Some(Box::new(double_output(ShapeFact::Scalar)));
    }
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

/// Decode an authored structural dimension without routing integer source
/// text through binary64 or an unchecked float-to-usize cast.
fn literal_dimension(value: &LiteralValue) -> Option<Result<usize, ()>> {
    match value {
        LiteralValue::Integer { text, .. } => Some(
            text.parse::<i128>()
                .ok()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or(()),
        ),
        LiteralValue::Number(_) | LiteralValue::Real { .. } => {
            let value = LiteralContext::numeric_from_literal(value)?;
            Some(
                (value.is_finite()
                    && value >= 1.0
                    && value.fract() == 0.0
                    && value < usize::MAX as f64)
                    .then_some(value as usize)
                    .ok_or(()),
            )
        }
        LiteralValue::Unknown | LiteralValue::Vector(_) => None,
        _ => Some(Err(())),
    }
}

fn selected_axis(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ReductionAxis {
    let mut axis = ReductionAxis::Default;
    let mut index = 2;
    while index < request.arguments.len() {
        let literal = request.literals.literal_string_at(index);
        if let Some(text) = literal {
            match OrderStatisticOption::parse(&text) {
                Some(OrderStatisticOption::All) => axis = ReductionAxis::All,
                Some(OrderStatisticOption::Method) => {
                    index += 1;
                    if index >= request.arguments.len() {
                        diagnostics.push(diagnostic(
                            "RM-CATALOG-ORDER-METHOD",
                            "Method needs a text value",
                            index - 1,
                        ));
                        break;
                    }
                    if let Some(value) = request.literals.literal_string_at(index) {
                        match OrderStatisticOption::parse(&value) {
                            Some(
                                OrderStatisticOption::Midpoint
                                | OrderStatisticOption::Inclusive
                                | OrderStatisticOption::Exclusive,
                            ) => {}
                            Some(OrderStatisticOption::Approximate) => {
                                diagnostics.push(diagnostic(
                                    "RM-CATALOG-ORDER-METHOD",
                                    "Approximate method requires unsupported tall arrays",
                                    index,
                                ))
                            }
                            _ => diagnostics.push(diagnostic(
                                "RM-CATALOG-ORDER-METHOD",
                                "Unsupported Method value",
                                index,
                            )),
                        }
                    }
                }
                Some(OrderStatisticOption::Approximate) => diagnostics.push(diagnostic(
                    "RM-CATALOG-ORDER-METHOD",
                    "Approximate method requires unsupported tall arrays",
                    index,
                )),
                Some(
                    OrderStatisticOption::IncludeNan
                    | OrderStatisticOption::OmitNan
                    | OrderStatisticOption::Midpoint
                    | OrderStatisticOption::Inclusive
                    | OrderStatisticOption::Exclusive,
                ) => {}
                None => diagnostics.push(diagnostic(
                    "RM-CATALOG-ORDER-OPTION",
                    "Unsupported order-statistics option",
                    index,
                )),
            }
        } else if matches!(
            request.arguments[index].kind,
            ValueKindFact::String | ValueKindFact::Character
        ) {
            // A runtime-computed option might select `all` and change the result shape.
            axis = ReductionAxis::Unknown;
        } else if real_numeric_or_unknown(&request.arguments[index]) {
            match request
                .literals
                .literal_args
                .get(index)
                .and_then(literal_dimension)
            {
                Some(Ok(dimension)) => axis = ReductionAxis::Dimension(dimension - 1),
                Some(Err(())) => diagnostics.push(diagnostic(
                    "RM-CATALOG-ORDER-DIM",
                    "Dimension must be a positive integer",
                    index,
                )),
                None => {
                    if let Some(dimensions) = request.literals.literal_vector_at(index) {
                        let mut axes = Vec::with_capacity(dimensions.len());
                        for dimension in &dimensions {
                            match literal_dimension(dimension) {
                                Some(Ok(dim)) => axes.push(dim - 1),
                                _ => {
                                    diagnostics.push(diagnostic(
                                        "RM-CATALOG-ORDER-DIM",
                                        "vecdim requires positive integer dimensions",
                                        index,
                                    ));
                                    axes.clear();
                                    break;
                                }
                            }
                        }
                        if axes.is_empty() && dimensions.is_empty() {
                            diagnostics.push(diagnostic(
                                "RM-CATALOG-ORDER-DIM",
                                "vecdim cannot be empty",
                                index,
                            ));
                        }
                        axes.sort_unstable();
                        if axes.windows(2).any(|pair| pair[0] == pair[1]) {
                            diagnostics.push(diagnostic(
                                "RM-CATALOG-ORDER-DIM",
                                "vecdim cannot repeat a dimension",
                                index,
                            ));
                        }
                        axis = if axes.is_empty() {
                            ReductionAxis::Unknown
                        } else {
                            ReductionAxis::Dimensions(axes)
                        };
                    } else {
                        axis = ReductionAxis::Unknown;
                    }
                }
            }
        } else {
            diagnostics.push(diagnostic(
                "RM-CATALOG-ORDER-OPTION",
                "Expected dimension or text option",
                index,
            ));
        }
        index += 1;
    }
    axis
}

pub(super) fn infer_quantile(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    unit: QuantileRequestUnit,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(diagnostic(
            "RM-CATALOG-ORDER-ARITY",
            "Data and requested probabilities are required",
            request.arguments.len(),
        ));
    }
    for index in 0..request.arguments.len().min(2) {
        if !real_numeric_or_unknown(&request.arguments[index]) {
            diagnostics.push(diagnostic(
                "RM-CATALOG-ORDER-NUMERIC",
                "Data and probabilities must be real numeric arrays",
                index,
            ));
        }
    }
    let mut count = request
        .arguments
        .get(1)
        .and_then(|value| value.shape.element_count());
    if count == Some(1) && unit == QuantileRequestUnit::Probability {
        if let Some(value) = request
            .literals
            .literal_args
            .get(1)
            .and_then(literal_dimension)
        {
            if let Ok(value) = value {
                if value > 1 {
                    count = Some(value);
                }
            }
        } else {
            // A runtime-computed scalar can mean either a probability or a count.
            count = None;
        }
    }
    let axis = selected_axis(request, &mut diagnostics);
    let shape = request
        .arguments
        .first()
        .map_or(ShapeFact::Unknown, |value| {
            reduced_shape(&value.shape, count, axis)
        });
    finish(
        entry,
        request,
        vec![double_output(shape)],
        diagnostics,
        false,
    )
}

pub(super) fn infer_tiedrank(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(diagnostic(
            "RM-CATALOG-TIEDRANK-ARITY",
            "tiedrank expects data and at most two mode flags",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !real_numeric_or_unknown(value))
    {
        diagnostics.push(diagnostic(
            "RM-CATALOG-TIEDRANK-NUMERIC",
            "tiedrank requires real numeric data",
            0,
        ));
    }
    let mut flags = [None; 2];
    let mut unknown_flag = false;
    for index in 1..request.arguments.len().min(3) {
        let argument = &request.arguments[index];
        if !real_numeric_or_unknown(argument)
            || argument
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
        {
            diagnostics.push(diagnostic(
                "RM-CATALOG-TIEDRANK-FLAG",
                "Mode flags must be scalar zero or one",
                index,
            ));
            continue;
        }
        match request.literals.numeric_at(index) {
            Some(0.0) => flags[index - 1] = Some(false),
            Some(1.0) => flags[index - 1] = Some(true),
            Some(_) => diagnostics.push(diagnostic(
                "RM-CATALOG-TIEDRANK-FLAG",
                "Mode flags must be zero or one",
                index,
            )),
            None => unknown_flag = true,
        }
    }
    let mode = if unknown_flag {
        None
    } else {
        TiedRankMode::from_flags(flags[0], flags[1])
    };
    if mode.is_none() && !unknown_flag {
        diagnostics.push(diagnostic(
            "RM-CATALOG-TIEDRANK-FLAG",
            "Unsupported mode flag combination",
            1,
        ));
    }
    let shape = request
        .arguments
        .first()
        .map_or(ShapeFact::Unknown, |value| {
            if value.shape.element_count() == Some(0) {
                ShapeFact::from(vec![Some(0), Some(1)])
            } else {
                value.shape.clone()
            }
        });
    let tie_shape = mode.map_or(ShapeFact::Unknown, |mode| {
        tie_adjustment_shape(&shape, mode)
    });
    finish(
        entry,
        request,
        vec![double_output(shape), double_output(tie_shape)],
        diagnostics,
        true,
    )
}
