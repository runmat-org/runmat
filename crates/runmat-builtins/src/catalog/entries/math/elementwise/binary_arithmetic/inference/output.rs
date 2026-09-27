use super::{admission::AdmittedCall, class, BinaryArithmeticInferencePolicy};
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::{
    BinaryArithmeticInferenceRule, BuiltinCatalogEntry, BuiltinInferenceRule, MathInferenceRule,
};
use runmat_types::{
    broadcast_shape, CallInference, CallRequest, ContiguityFact, DynamicReason, LayoutFact,
    NumericDomain, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
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
    // Device provenance and the eventual automatic placement decision are not
    // represented in ValueFact. Input residency alone cannot promise residency
    // after provider fallback or automatic preparation.
    let floating_pair = [left, right].iter().all(|fact| {
        matches!(fact.storage, StorageFact::Dense | StorageFact::Unknown)
            && fact.numeric().is_some_and(|n| {
                n.class.integer_class().is_none() && n.domain == NumericDomain::Real
            })
    });
    output.residency = if !floating_pair
        && left.residency == ResidencyFact::Host
        && right.residency == ResidencyFact::Host
    {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
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
    if left.storage == StorageFact::Sparse || right.storage == StorageFact::Sparse {
        let preserves_sparse = matches!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::BinaryArithmetic(
                BinaryArithmeticInferenceRule::Add
                    | BinaryArithmeticInferenceRule::Subtract
                    | BinaryArithmeticInferenceRule::Multiply
            ))
        ) && left.storage == StorageFact::Sparse
            && right.storage == StorageFact::Sparse
            && left.shape.element_count().is_some()
            && left.shape.is_proven_equivalent(&right.shape);
        output.storage = if preserves_sparse {
            StorageFact::Sparse
        } else {
            StorageFact::Unknown
        };
        // Sparse/dense selection can depend on operand values (for example,
        // adding zero versus a nonzero scalar to implicit sparse zeros).
        output.layout = LayoutFact::Unknown;
        output.contiguity = ContiguityFact::Unknown;
    }
    apply_prototype(admitted.prototype, &mut output, &mut diagnostics);
    if left.storage == StorageFact::Unknown || right.storage == StorageFact::Unknown {
        output.storage = StorageFact::Unknown;
    }
    if output.storage == StorageFact::Scalar
        && output
            .numeric()
            .is_some_and(|n| n.class == runmat_types::NumericClass::Single)
    {
        output.storage = StorageFact::Dense;
    }
    match output.residency {
        ResidencyFact::Host => {}
        ResidencyFact::Device { .. } => output.storage = StorageFact::Opaque,
        _ => output.storage = StorageFact::Unknown,
    }
    if matches!(output.storage, StorageFact::Unknown | StorageFact::Opaque) {
        output.layout = LayoutFact::Unknown;
        output.contiguity = ContiguityFact::Unknown;
    }
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
