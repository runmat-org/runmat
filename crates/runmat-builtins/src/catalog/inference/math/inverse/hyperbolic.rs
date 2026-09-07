use super::inverse_hyperbolic_literal_domain;
use crate::catalog::inference::support::facts::{materialize, preserve_shape_as_dynamic};
use crate::catalog::inference::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, InverseHyperbolicFunction};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    StorageFact, ValueFact, ValueKindFact,
};

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
