use super::{apply_inverse_trigonometric_like, inverse_trigonometric_literal_domain};
use crate::catalog::inference::support::facts::{materialize, preserve_shape_as_dynamic};
use crate::catalog::inference::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, InverseTrigonometricFunction};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    StorageFact, ValueFact, ValueKindFact,
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
