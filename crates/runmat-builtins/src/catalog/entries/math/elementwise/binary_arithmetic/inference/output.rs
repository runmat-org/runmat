use super::{admission::AdmittedCall, class, BinaryArithmeticInferencePolicy};
use crate::catalog::inference::{argument_error, finish_fixed, preserved_binary_residency};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    broadcast_shape, CallInference, CallRequest, DynamicReason, NumericDomain, ResidencyFact,
    ShapeFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(
    policy: BinaryArithmeticInferencePolicy,
    admitted: AdmittedCall<'_>,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = admitted.diagnostics;
    let (Some(left), Some(right)) = (admitted.left, admitted.right) else {
        return finish_fixed(entry, request, dynamic(), diagnostics);
    };

    let mut output = dynamic();
    output.shape = match broadcast_shape(&left.shape, &right.shape) {
        Ok(shape) => shape,
        Err(diagnostic) => {
            diagnostics.push(diagnostic);
            ShapeFact::Unknown
        }
    };
    output.residency = preserved_binary_residency(&left.residency, &right.residency);
    match class::result(policy, left, right) {
        Ok(kind) => output.kind = kind,
        Err(DynamicReason::UnsupportedRepresentation) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-BINARY-ARITHMETIC-INPUT",
                "operands do not form a supported arithmetic pair",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        Err(reason) => output.certainty = runmat_types::CertaintyFact::Dynamic(reason),
    }
    crate::catalog::inference::materialize(&mut output);
    apply_prototype(admitted.prototype, &mut output, &mut diagnostics);
    finish_fixed(entry, request, output, diagnostics)
}

fn apply_prototype(
    prototype: Option<&ValueFact>,
    output: &mut ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    let Some(prototype) = prototype else { return };
    output.residency = match &prototype.residency {
        ResidencyFact::Host => ResidencyFact::Host,
        ResidencyFact::Device { provider } => ResidencyFact::Device {
            provider: provider.clone(),
        },
        _ => ResidencyFact::Unknown,
    };
    if matches!(
        prototype.numeric().map(|numeric| numeric.domain),
        Some(NumericDomain::Complex)
    ) {
        if let ValueKindFact::Numeric(numeric) = &mut output.kind {
            numeric.domain = NumericDomain::Complex;
        }
    } else if !matches!(
        prototype.kind,
        ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BINARY-ARITHMETIC-PROTOTYPE",
            "the 'like' prototype must be numeric, logical, character, or provider-resident numeric data",
            3,
        ));
    }
}

fn dynamic() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}
