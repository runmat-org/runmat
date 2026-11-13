use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, InferenceDiagnostic, LiteralValue,
    NumericClass, NumericDomain, NumericFact, ObjectFact, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let args = &request.arguments;
    let text_first = args
        .first()
        .is_some_and(|value| is_text_scalar_fact(value, false));
    let shorthand = text_first
        || (args.len() <= 2
            && args.first().is_some_and(|value| {
                matches!(value.kind, ValueKindFact::Unknown) && is_text_scalar_fact(value, true)
            }));
    let mut numerator = coefficient_fact(None);
    let mut denominator = coefficient_fact(None);
    if args.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-ARITY",
            "tf requires a variable or numerator and denominator",
            0,
        ));
    } else if text_first {
        if args.len() > 2 {
            diagnostics.push(argument_error(
                "RM-CATALOG-TF-ARITY",
                "variable shorthand accepts at most one sample time",
                2,
            ));
        }
        if let Some(name) = request.literals.literal_string_at(0) {
            if !is_variable(&name) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-TF-VARIABLE",
                    "unsupported transfer-function variable",
                    0,
                ));
            }
            if matches!(name.as_str(), "s" | "p" | "z" | "q" | "z^-1" | "q^-1") {
                numerator = coefficient_fact(Some((Some(2), Some(NumericDomain::Real))));
                denominator = coefficient_fact(Some((Some(1), Some(NumericDomain::Real))));
            }
        }
        if let Some(sample_time) = args.get(1) {
            validate_sample_time(
                sample_time,
                request.literals.literal_args.get(1),
                1,
                &mut diagnostics,
            );
        }
    } else if args.len() >= 2 {
        numerator = coefficient_fact(coefficients(&args[0], 0, &mut diagnostics));
        denominator = coefficient_fact(coefficients(&args[1], 1, &mut diagnostics));
        validate_options(request, &mut diagnostics);
    } else if !shorthand {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-ARITY",
            "numeric constructor requires a denominator",
            1,
        ));
    }
    let mut properties = BTreeMap::new();
    properties.insert("Numerator".into(), numerator);
    properties.insert("Denominator".into(), denominator);
    properties.insert(
        "Variable".into(),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
    );
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    for name in ["Ts", "InputDelay", "OutputDelay"] {
        properties.insert(name.into(), scalar.clone());
    }
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TRANSFER_FUNCTION.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, output, diagnostics)
}

fn coefficients(
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<(Option<usize>, Option<NumericDomain>)> {
    if value.storage == StorageFact::Sparse {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-COEFFICIENTS",
            "sparse coefficient arrays are unsupported",
            index,
        ));
        return None;
    }
    let domain = match &value.kind {
        ValueKindFact::Numeric(numeric) => Some(numeric.domain),
        ValueKindFact::Logical => Some(NumericDomain::Real),
        ValueKindFact::Unknown => None,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TF-COEFFICIENTS",
                "coefficients must be numeric or logical scalars or vectors",
                index,
            ));
            return None;
        }
    };
    let Some(dims) = value.shape.known_dims() else {
        return Some((None, domain));
    };
    if dims.len() > 2
        || dims.len() == 2
            && dims[0]
                .zip(dims[1])
                .is_some_and(|(rows, cols)| rows > 1 && cols > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-COEFFICIENTS",
            "coefficients must be scalars or vectors",
            index,
        ));
        return None;
    }
    // Leading zeros are stripped at runtime, so a vector's stored length is not
    // known without its values. A scalar remains exactly one coefficient.
    Some((
        (value.shape.element_count() == Some(1)).then_some(1),
        domain,
    ))
}

fn coefficient_fact(known: Option<(Option<usize>, Option<NumericDomain>)>) -> ValueFact {
    let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
    value.shape = ShapeFact::from(vec![Some(1), known.and_then(|v| v.0)]);
    value.storage = StorageFact::Dense;
    value.residency = ResidencyFact::Host;
    if let Some((_, Some(domain))) = known {
        value.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain,
        });
    }
    value
}

fn validate_options(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let rest = request.arguments.get(2..).unwrap_or(&[]);
    if rest.len() == 1 {
        validate_sample_time(
            &rest[0],
            request.literals.literal_args.get(2),
            2,
            diagnostics,
        );
        return;
    }
    if !rest.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-OPTIONS",
            "options require a scalar sample time or name/value pairs",
            2,
        ));
        return;
    }
    for (pair_index, pair) in rest.chunks_exact(2).enumerate() {
        let index = 2 + pair_index * 2;
        if !is_text_scalar_fact(&pair[0], true) {
            diagnostics.push(argument_error(
                "RM-CATALOG-TF-OPTION",
                "option names must be text",
                index,
            ));
        }
        match request
            .literals
            .literal_string_at(index)
            .as_deref()
            .map(str::trim)
        {
            Some("variable") => validate_variable(
                &pair[1],
                request.literals.literal_string_at(index + 1).as_deref(),
                index + 1,
                diagnostics,
            ),
            Some("ts" | "sampletime") => validate_sample_time(
                &pair[1],
                request.literals.literal_args.get(index + 1),
                index + 1,
                diagnostics,
            ),
            Some(_) => diagnostics.push(argument_error(
                "RM-CATALOG-TF-OPTION",
                "unsupported tf option",
                index,
            )),
            None => {}
        }
    }
}

fn validate_variable(
    value: &ValueFact,
    literal: Option<&str>,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !is_text_scalar_fact(value, true) || literal.is_some_and(|name| !is_variable(name.trim())) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-VARIABLE",
            "Variable must be a supported control variable",
            index,
        ));
    }
}

fn is_text_scalar_fact(value: &ValueFact, allow_unknown_kind: bool) -> bool {
    match value.kind {
        ValueKindFact::Unknown => allow_unknown_kind && possible_text_shape(value),
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

fn is_variable(value: &str) -> bool {
    matches!(value, "s" | "p" | "z" | "q" | "z^-1" | "q^-1")
}

fn validate_sample_time(
    value: &ValueFact,
    literal: Option<&LiteralValue>,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !matches!(value.kind, ValueKindFact::Unknown)
        && (!(value
            .numeric()
            .is_some_and(|n| n.domain == NumericDomain::Real)
            || matches!(value.kind, ValueKindFact::Logical))
            || value.shape.element_count().is_some_and(|count| count != 1))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-SAMPLE-TIME",
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
    if known.is_some_and(|value| !value.is_finite() || value < 0.0 && value != -1.0) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TF-SAMPLE-TIME",
            "Ts must be 0, positive, or -1",
            index,
        ));
    }
}
