use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, LiteralValue, NumericDomain, NumericFact, ResidencyFact, ValueFact, ValueKindFact,
};

pub(super) fn apply_inverse_trigonometric_like(
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
