use super::representation::OutputRepresentation;
use crate::catalog::inference::{argument_error, literal_text};
use runmat_types::{
    standard, CallRequest, ClassIdentity, InferenceDiagnostic, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ValueFact, ValueKindFact,
};

pub(super) fn from_literal(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<OutputRepresentation> {
    let selector = request.arguments.get(1)?;
    let Some(text) = request.literals.literal_args.get(1).and_then(literal_text) else {
        if !matches!(
            selector.kind,
            ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-TYPECAST-SELECTOR",
                "typecast output class must be a character or string scalar",
                1,
            ));
        }
        return None;
    };
    let Ok(identity) = ClassIdentity::new(text.to_ascii_lowercase()) else {
        diagnostics.push(argument_error(
            "RM-CATALOG-TYPECAST-CLASS",
            "typecast output class is not a valid class identity",
            1,
        ));
        return None;
    };
    if let Some(class) = NumericClass::from_class_identity(&identity) {
        return Some(OutputRepresentation::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }));
    }
    if identity.is(standard::LOGICAL) {
        return Some(OutputRepresentation::Logical);
    }
    diagnostics.push(argument_error(
        "RM-CATALOG-TYPECAST-CLASS",
        if identity.is(standard::CHAR) {
            "typecast character output is not currently supported"
        } else {
            "typecast output class is unsupported"
        },
        1,
    ));
    None
}

pub(super) fn from_prototype(
    request: &CallRequest,
    source: &ValueFact,
    prototype: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<OutputRepresentation> {
    match request.literals.literal_args.get(1).and_then(literal_text) {
        Some(keyword) if keyword.eq_ignore_ascii_case("like") => {}
        Some(_) => diagnostics.push(argument_error(
            "RM-CATALOG-TYPECAST-LIKE",
            "the three-input typecast form requires the literal string \"like\"",
            1,
        )),
        None if !matches!(
            request.arguments.get(1)?.kind,
            ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown
        ) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-TYPECAST-LIKE",
                "the three-input typecast form requires a text selector",
                1,
            ))
        }
        None => return None,
    }
    validate_host_values(source, prototype, diagnostics);
    match prototype.kind {
        ValueKindFact::Numeric(fact) => Some(OutputRepresentation::Numeric(fact)),
        ValueKindFact::Logical => Some(OutputRepresentation::Logical),
        ValueKindFact::Unknown => None,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TYPECAST-PROTOTYPE",
                "typecast requires a numeric or logical prototype",
                2,
            ));
            None
        }
    }
}

fn validate_host_values(
    source: &ValueFact,
    prototype: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if matches!(
        source.residency,
        ResidencyFact::Host | ResidencyFact::Unknown
    ) && matches!(
        prototype.residency,
        ResidencyFact::Host | ResidencyFact::Unknown
    ) {
        return;
    }
    diagnostics.push(argument_error(
        "RM-CATALOG-TYPECAST-LIKE-RESIDENCY",
        "typecast like prototypes and their source must be host values",
        if matches!(
            source.residency,
            ResidencyFact::Host | ResidencyFact::Unknown
        ) {
            2
        } else {
            0
        },
    ));
}
