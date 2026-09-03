use super::support::facts::{materialize, preserve_shape_as_dynamic};
use super::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, InverseHyperbolicFunction, InverseTrigonometricFunction};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer_inverse_trigonometric(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: InverseTrigonometricFunction,
) -> CallInference {
    let name = match function {
        InverseTrigonometricFunction::Sine => "asin",
        InverseTrigonometricFunction::Cosine => "acos",
        InverseTrigonometricFunction::Tangent => "atan",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-TRIGONOMETRIC-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let accepts_like = function == InverseTrigonometricFunction::Tangent;
    let arity_is_valid = if accepts_like {
        matches!(request.arguments.len(), 1 | 3)
    } else {
        request.arguments.len() == 1
    };
    if !arity_is_valid {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-TRIGONOMETRIC-ARITY",
            if accepts_like {
                format!("{name} accepts one input or an input followed by \"like\" and a prototype")
            } else {
                format!("{name} accepts exactly one input")
            },
            request.arguments.len().saturating_sub(1),
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-TRIGONOMETRIC-SPARSE",
            format!("{name} does not currently accept sparse input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let literal_domain = request
        .literals
        .literal_args
        .first()
        .and_then(|literal| inverse_trigonometric_literal_domain(literal, function));
    let mut output = input.clone();
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-TRIGONOMETRIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => {
            output.kind = numeric_kind(*class, NumericDomain::Complex);
            materialize(&mut output);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            let output_class = if matches!(class, NumericClass::Double | NumericClass::Single) {
                *class
            } else {
                NumericClass::Double
            };
            if let Some(domain) = literal_domain.or_else(|| {
                (function == InverseTrigonometricFunction::Tangent).then_some(NumericDomain::Real)
            }) {
                output.kind = numeric_kind(output_class, domain);
                materialize(&mut output);
            } else {
                preserve_shape_as_dynamic(&mut output);
                output.residency = input.residency.clone();
            }
        }
        ValueKindFact::Logical => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize(&mut output);
        }
        ValueKindFact::Character => {
            if let Some(domain) = literal_domain.or_else(|| {
                (function == InverseTrigonometricFunction::Tangent).then_some(NumericDomain::Real)
            }) {
                output.kind = numeric_kind(NumericClass::Double, domain);
                materialize(&mut output);
            } else {
                preserve_shape_as_dynamic(&mut output);
                output.residency = input.residency.clone();
            }
        }
        ValueKindFact::Unknown => {
            preserve_shape_as_dynamic(&mut output);
            output.residency = input.residency.clone();
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-TRIGONOMETRIC-INPUT",
                format!("{name} requires numeric, logical, or character input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    if accepts_like {
        apply_inverse_trigonometric_like(request, &mut output, name, &mut diagnostics);
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_inverse_hyperbolic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: InverseHyperbolicFunction,
) -> CallInference {
    let name = match function {
        InverseHyperbolicFunction::Cosine => "acosh",
        InverseHyperbolicFunction::Sine => "asinh",
        InverseHyperbolicFunction::Tangent => "atanh",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-HYPERBOLIC-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-HYPERBOLIC-ARITY",
            format!("{name} accepts exactly one input"),
            request.arguments.len().saturating_sub(1),
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-INVERSE-HYPERBOLIC-SPARSE",
            format!("{name} does not currently accept sparse input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let literal_domain = request
        .literals
        .literal_args
        .first()
        .and_then(|literal| inverse_hyperbolic_literal_domain(literal, function));
    let always_real = function == InverseHyperbolicFunction::Sine;
    let logical_always_real = matches!(
        function,
        InverseHyperbolicFunction::Sine | InverseHyperbolicFunction::Tangent
    );
    let mut output = input.clone();
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-HYPERBOLIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => {
            output.kind = numeric_kind(*class, NumericDomain::Complex);
            materialize(&mut output);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            let output_class = if matches!(class, NumericClass::Double | NumericClass::Single) {
                *class
            } else {
                NumericClass::Double
            };
            if let Some(domain) =
                literal_domain.or_else(|| always_real.then_some(NumericDomain::Real))
            {
                output.kind = numeric_kind(output_class, domain);
                materialize(&mut output);
            } else {
                preserve_shape_as_dynamic(&mut output);
                output.residency = input.residency.clone();
            }
        }
        ValueKindFact::Logical => {
            if let Some(domain) =
                literal_domain.or_else(|| logical_always_real.then_some(NumericDomain::Real))
            {
                output.kind = numeric_kind(NumericClass::Double, domain);
                materialize(&mut output);
            } else {
                preserve_shape_as_dynamic(&mut output);
                output.residency = input.residency.clone();
            }
        }
        ValueKindFact::Character => {
            if let Some(domain) =
                literal_domain.or_else(|| always_real.then_some(NumericDomain::Real))
            {
                output.kind = numeric_kind(NumericClass::Double, domain);
                materialize(&mut output);
            } else {
                preserve_shape_as_dynamic(&mut output);
                output.residency = input.residency.clone();
            }
        }
        ValueKindFact::Unknown => {
            preserve_shape_as_dynamic(&mut output);
            output.residency = input.residency.clone();
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-HYPERBOLIC-INPUT",
                format!("{name} requires numeric, logical, or character input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn inverse_hyperbolic_literal_domain(
    literal: &LiteralValue,
    function: InverseHyperbolicFunction,
) -> Option<NumericDomain> {
    match literal {
        LiteralValue::Number(value) => Some(inverse_hyperbolic_real_domain(*value, function)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => text
            .parse::<f64>()
            .ok()
            .map(|value| inverse_hyperbolic_real_domain(value, function)),
        LiteralValue::Bool(value) => Some(inverse_hyperbolic_real_domain(
            if *value { 1.0 } else { 0.0 },
            function,
        )),
        LiteralValue::Character(value) => combine_numeric_domains(value.chars().map(|character| {
            inverse_hyperbolic_real_domain(f64::from(u32::from(character)), function)
        })),
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Vector(values) => combine_optional_numeric_domains(
            values
                .iter()
                .map(|value| inverse_hyperbolic_literal_domain(value, function)),
        ),
        LiteralValue::Matrix(rows) => combine_optional_numeric_domains(
            rows.iter()
                .flatten()
                .map(|value| inverse_hyperbolic_literal_domain(value, function)),
        ),
        LiteralValue::Empty => Some(NumericDomain::Real),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn inverse_hyperbolic_real_domain(
    value: f64,
    function: InverseHyperbolicFunction,
) -> NumericDomain {
    let is_real = match function {
        InverseHyperbolicFunction::Cosine => value.is_nan() || value >= 1.0,
        InverseHyperbolicFunction::Sine => true,
        InverseHyperbolicFunction::Tangent => value.is_nan() || (-1.0..=1.0).contains(&value),
    };
    if is_real {
        NumericDomain::Real
    } else {
        NumericDomain::Complex
    }
}

fn combine_optional_numeric_domains(
    domains: impl IntoIterator<Item = Option<NumericDomain>>,
) -> Option<NumericDomain> {
    let mut combined = NumericDomain::Real;
    for domain in domains {
        if domain? == NumericDomain::Complex {
            combined = NumericDomain::Complex;
        }
    }
    Some(combined)
}

fn combine_numeric_domains(
    domains: impl IntoIterator<Item = NumericDomain>,
) -> Option<NumericDomain> {
    combine_optional_numeric_domains(domains.into_iter().map(Some))
}

fn apply_inverse_trigonometric_like(
    request: &CallRequest,
    output: &mut ValueFact,
    name: &str,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    let [_, keyword, prototype] = request.arguments.as_slice() else {
        return;
    };
    let keyword_literal = request
        .literals
        .literal_args
        .get(1)
        .and_then(|literal| match literal {
            LiteralValue::String(value)
            | LiteralValue::Character(value)
            | LiteralValue::Keyword(value) => Some(value.as_str()),
            _ => None,
        });
    match keyword_literal {
        Some(value) if value.eq_ignore_ascii_case("like") => {}
        Some(_) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-TRIGONOMETRIC-LIKE",
                format!("{name} accepts only the \"like\" option"),
                1,
            ));
            output.residency = ResidencyFact::Unknown;
            return;
        }
        None if matches!(
            keyword.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) =>
        {
            output.residency = ResidencyFact::Unknown;
            return;
        }
        None => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-TRIGONOMETRIC-LIKE",
                format!("{name} requires \"like\" as its second input"),
                1,
            ));
            output.residency = ResidencyFact::Unknown;
            return;
        }
    }

    let output_is_complex = matches!(
        output.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        })
    );
    match &prototype.kind {
        ValueKindFact::Numeric(prototype_numeric) => {
            if prototype_numeric.domain == NumericDomain::Complex {
                if let ValueKindFact::Numeric(output_numeric) = &mut output.kind {
                    output_numeric.domain = NumericDomain::Complex;
                }
            } else if output_is_complex {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INVERSE-TRIGONOMETRIC-PROTOTYPE",
                    format!("{name} cannot place a complex result like a real prototype"),
                    2,
                ));
            }
            output.residency = prototype.residency.clone();
        }
        ValueKindFact::Logical => {
            if output_is_complex {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INVERSE-TRIGONOMETRIC-PROTOTYPE",
                    format!("{name} cannot place a complex result like a real prototype"),
                    2,
                ));
            }
            output.residency = prototype.residency.clone();
        }
        ValueKindFact::Unknown => {
            output.residency = prototype.residency.clone();
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INVERSE-TRIGONOMETRIC-PROTOTYPE",
                format!("{name} requires a numeric or logical prototype"),
                2,
            ));
            output.residency = ResidencyFact::Unknown;
        }
    }
}

fn inverse_trigonometric_literal_domain(
    literal: &LiteralValue,
    function: InverseTrigonometricFunction,
) -> Option<NumericDomain> {
    if function == InverseTrigonometricFunction::Tangent {
        return literal_is_real(literal).then_some(NumericDomain::Real);
    }
    match literal {
        LiteralValue::Number(value) => Some(unit_interval_domain(*value)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse::<f64>().ok().map(unit_interval_domain)
        }
        LiteralValue::Bool(_) => Some(NumericDomain::Real),
        LiteralValue::Character(value) => Some(if value.chars().all(|ch| u32::from(ch) <= 1) {
            NumericDomain::Real
        } else {
            NumericDomain::Complex
        }),
        LiteralValue::Complex { .. } => Some(NumericDomain::Complex),
        LiteralValue::Vector(values) => combine_inverse_literal_domains(values, function),
        LiteralValue::Matrix(rows) => {
            let values = rows.iter().flatten().cloned().collect::<Vec<_>>();
            combine_inverse_literal_domains(&values, function)
        }
        LiteralValue::Empty => Some(NumericDomain::Real),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn literal_is_real(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Number(_)
        | LiteralValue::Real { .. }
        | LiteralValue::Integer { .. }
        | LiteralValue::Bool(_)
        | LiteralValue::Character(_)
        | LiteralValue::Empty => true,
        LiteralValue::Vector(values) => values.iter().all(literal_is_real),
        LiteralValue::Matrix(rows) => rows.iter().flatten().all(literal_is_real),
        LiteralValue::Complex { .. }
        | LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => false,
    }
}

fn unit_interval_domain(value: f64) -> NumericDomain {
    if value.is_nan() || (-1.0..=1.0).contains(&value) {
        NumericDomain::Real
    } else {
        NumericDomain::Complex
    }
}

fn combine_inverse_literal_domains(
    values: &[LiteralValue],
    function: InverseTrigonometricFunction,
) -> Option<NumericDomain> {
    let mut domain = NumericDomain::Real;
    for value in values {
        let value_domain = inverse_trigonometric_literal_domain(value, function)?;
        if value_domain == NumericDomain::Complex {
            domain = NumericDomain::Complex;
        }
    }
    Some(domain)
}
