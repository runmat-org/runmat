use super::super::fitted_distribution_model::FittedDistributionFamily;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let arguments = &request.arguments;
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    let Some(distribution) = arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-RANDOM-ARITY",
            "random requires a fitted object or distribution name",
            0,
        ));
        return finish_fixed(entry, request, output, diagnostics);
    };
    let first_size = match &distribution.kind {
        ValueKindFact::Object(object) => {
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| !class.is(standard::PROBABILITY_DISTRIBUTION))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RANDOM-OBJECT",
                    "expected a fitted ProbabilityDistribution object",
                    0,
                ));
            }
            Some(1)
        }
        ValueKindFact::String | ValueKindFact::Character => request
            .literals
            .literal_string_at(0)
            .and_then(|name| match FittedDistributionFamily::from_keyword(&name) {
                Some(family) => Some(1 + family.parameter_names().len()),
                None => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-RANDOM-FAMILY",
                        "unsupported distribution family",
                        0,
                    ));
                    None
                }
            }),
        ValueKindFact::Unknown => None,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-RANDOM-FAMILY",
                "expected a fitted object or distribution name",
                0,
            ));
            None
        }
    };
    if let Some(first_size) = first_size {
        if arguments.len() < first_size {
            diagnostics.push(argument_error(
                "RM-CATALOG-RANDOM-ARITY",
                "named family parameters are required",
                arguments.len().saturating_sub(1),
            ));
        }
        for (index, parameter) in arguments.iter().enumerate().take(first_size).skip(1) {
            if !matches!(
                parameter.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            ) || parameter.storage == StorageFact::Sparse
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RANDOM-PARAMETER",
                    "named parameters must be real numeric or logical scalars",
                    index,
                ));
            }
            if parameter
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RANDOM-PARAMETER",
                    "named parameters must be scalar",
                    index,
                ));
            }
        }
        let sizes = &arguments[first_size.min(arguments.len())..];
        for (offset, size) in sizes.iter().enumerate() {
            if !matches!(
                size.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            ) || size.storage == StorageFact::Sparse
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RANDOM-SIZE",
                    "sizes must be numeric scalar or vector controls",
                    first_size + offset,
                ));
            }
        }
        output.shape = if sizes.is_empty() {
            ShapeFact::Scalar
        } else {
            infer_size(request, first_size, &mut diagnostics)
        };
        let resident_parameter = arguments
            .iter()
            .take(first_size)
            .skip(1)
            .any(|parameter| parameter.residency != ResidencyFact::Host);
        output.residency = if resident_parameter {
            ResidencyFact::Unknown
        } else {
            ResidencyFact::Host
        };
        output.storage = match (output.shape.element_count(), resident_parameter) {
            (Some(1), false) => StorageFact::Scalar,
            (Some(_), false) => StorageFact::Dense,
            _ => StorageFact::Unknown,
        };
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn infer_size(
    request: &CallRequest,
    start: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ShapeFact {
    let mut dimensions = Vec::new();
    for index in start..request.arguments.len() {
        let Some(literal) = request.literals.literal_args.get(index) else {
            return ShapeFact::Unknown;
        };
        let values = match literal {
            LiteralValue::Vector(values) => values.iter().collect::<Vec<_>>(),
            LiteralValue::Matrix(rows) if rows.len() == 1 => rows[0].iter().collect(),
            LiteralValue::Matrix(rows) if rows.iter().all(|row| row.len() == 1) => {
                rows.iter().flatten().collect()
            }
            LiteralValue::Unknown => return ShapeFact::Unknown,
            LiteralValue::Matrix(_) => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RANDOM-SIZE",
                    "size must be a scalar or vector",
                    index,
                ));
                return ShapeFact::Unknown;
            }
            other => vec![other],
        };
        for value in values {
            let extent = match value {
                LiteralValue::Integer { text, .. } => text.parse::<usize>().ok(),
                LiteralValue::Number(_) | LiteralValue::Real { .. } => {
                    runmat_types::LiteralContext::numeric_from_literal(value)
                        .filter(|extent| {
                            extent.is_finite()
                                && *extent >= 0.0
                                && extent.fract() == 0.0
                                && *extent < usize::MAX as f64
                        })
                        .map(|extent| extent as usize)
                }
                LiteralValue::Unknown => return ShapeFact::Unknown,
                _ => None,
            };
            let Some(extent) = extent else {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RANDOM-SIZE",
                    "size extents must be finite nonnegative integers",
                    index,
                ));
                return ShapeFact::Unknown;
            };
            dimensions.push(Some(extent));
        }
    }
    if dimensions.is_empty() {
        dimensions = vec![Some(0), Some(0)];
    }
    if dimensions.len() == 1 {
        dimensions.push(dimensions[0]);
    }
    let extent_values = dimensions.iter().copied().flatten().collect::<Vec<_>>();
    if !extent_values.contains(&0)
        && extent_values
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .is_none()
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-RANDOM-SIZE",
            "output shape exceeds structural range",
            start,
        ));
        return ShapeFact::Unknown;
    }
    ShapeFact::from(dimensions)
}
