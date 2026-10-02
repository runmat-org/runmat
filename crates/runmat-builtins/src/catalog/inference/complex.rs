use super::*;

pub(super) fn infer_complex(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.as_slice() {
        [input] => unary(input, &mut diagnostics),
        [real, imag] => binary(real, imag, &mut diagnostics),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-COMPLEX-ARITY",
                "complex requires one or two inputs",
                0,
            ));
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn numeric(input: &ValueFact) -> Option<NumericFact> {
    match input.kind {
        ValueKindFact::Numeric(fact) => Some(fact),
        ValueKindFact::Logical => Some(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => None,
    }
}

fn invalid(
    diagnostics: &mut Vec<InferenceDiagnostic>,
    message: &str,
    argument: usize,
) -> ValueFact {
    diagnostics.push(argument_error(
        "RM-CATALOG-COMPLEX-INPUT",
        message,
        argument,
    ));
    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
}

fn unary(input: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) -> ValueFact {
    if matches!(input.kind, ValueKindFact::Unknown) {
        return input.clone();
    }
    let Some(mut kind) = numeric(input) else {
        return invalid(diagnostics, "complex requires numeric input", 0);
    };
    kind.domain = NumericDomain::Complex;
    let mut output = input.clone();
    output.kind = ValueKindFact::Numeric(kind);
    if !matches!(input.residency, ResidencyFact::Host) {
        // Provider composition may fall back to host gathering.
        output.residency = ResidencyFact::Unknown;
    }
    output.storage = if input.shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output
}

fn binary(
    real: &ValueFact,
    imag: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    if matches!(real.kind, ValueKindFact::Unknown) || matches!(imag.kind, ValueKindFact::Unknown) {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    let Some(a) = numeric(real) else {
        return invalid(diagnostics, "complex components must be real numeric", 0);
    };
    let Some(b) = numeric(imag) else {
        return invalid(diagnostics, "complex components must be real numeric", 1);
    };
    if a.domain != NumericDomain::Real || b.domain != NumericDomain::Real {
        return invalid(diagnostics, "binary complex components must be real", 0);
    }
    let integer = |class| !matches!(class, NumericClass::Single | NumericClass::Double);
    let scalar_double = |input: &ValueFact, kind: NumericFact| {
        kind.class == NumericClass::Double
            && input.shape.element_count() == Some(1)
            && matches!(input.storage, StorageFact::Scalar | StorageFact::Dense)
    };
    let class = if a.class == b.class || (integer(a.class) && scalar_double(imag, b)) {
        a.class
    } else if integer(b.class) && scalar_double(real, a) {
        b.class
    } else if integer(a.class) || integer(b.class) {
        return invalid(
            diagnostics,
            "integer components require matching classes or a full scalar double",
            1,
        );
    } else if a.class == NumericClass::Single || b.class == NumericClass::Single {
        NumericClass::Single
    } else {
        NumericClass::Double
    };
    let shape = if real.shape.element_count() == Some(1) {
        imag.shape.clone()
    } else if imag.shape.element_count() == Some(1) || real.shape.is_proven_equivalent(&imag.shape)
    {
        real.shape.clone()
    } else if real.shape.element_count().is_some() && imag.shape.element_count().is_some() {
        return invalid(
            diagnostics,
            "complex component sizes must match unless one is scalar",
            1,
        );
    } else {
        ShapeFact::Unknown
    };
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    let mut output = ValueFact::proven(numeric_kind(class, NumericDomain::Complex), shape, storage);
    output.residency = if matches!(real.residency, ResidencyFact::Host)
        && matches!(imag.residency, ResidencyFact::Host)
    {
        ResidencyFact::Host
    } else {
        // Mixed ownership/provider facts require runtime resolution.
        ResidencyFact::Unknown
    };
    output
}
