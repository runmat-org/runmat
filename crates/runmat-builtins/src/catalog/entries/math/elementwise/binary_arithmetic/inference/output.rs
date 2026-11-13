use super::policy::SparseStorageRule;
mod prototype;
use super::{admission::AdmittedCall, class, BinaryArithmeticInferencePolicy, DiagnosticCollector};
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use prototype::apply_prototype;
use runmat_types::{
    broadcast_shape_checked, CallInference, CallRequest, CertaintyFact, ContiguityFact,
    DynamicReason, LayoutFact, NumericDomain, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(
    policy: BinaryArithmeticInferencePolicy,
    admitted: AdmittedCall<'_>,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let (output, diagnostics) = infer_fact(policy, admitted, request);
    finish_fixed(entry, request, output, diagnostics.messages)
}

pub(super) fn infer_fact(
    policy: BinaryArithmeticInferencePolicy,
    admitted: AdmittedCall<'_>,
    request: &CallRequest,
) -> (ValueFact, DiagnosticCollector) {
    let mut diagnostics = admitted.diagnostics;
    let (Some(left), Some(right)) = (admitted.left, admitted.right) else {
        return (dynamic(), diagnostics);
    };

    let mut output = dynamic();
    output.shape = match broadcast_shape_checked(&left.shape, &right.shape) {
        Ok(shape) => shape,
        Err(diagnostic) => {
            diagnostics.push(|| diagnostic.into_diagnostic());
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
        Ok(kind) => {
            output.certainty = if kind == ValueKindFact::Symbolic {
                CertaintyFact::Symbolic
            } else if left.certainty == CertaintyFact::Proven
                && right.certainty == CertaintyFact::Proven
            {
                // Class/domain inference has succeeded for proven operands.
                // Unknown eventual placement/storage is a separate fact, not
                // a reason to erase this semantic evidence from later stages.
                CertaintyFact::Proven
            } else {
                [left, right]
                    .iter()
                    .find_map(|input| match &input.certainty {
                        CertaintyFact::Dynamic(reason) => {
                            Some(CertaintyFact::Dynamic(reason.clone()))
                        }
                        _ => None,
                    })
                    .unwrap_or(CertaintyFact::Dynamic(DynamicReason::RuntimeValue))
            };
            output.kind = kind;
        }
        Err(DynamicReason::UnsupportedRepresentation) => {
            diagnostics.push(|| {
                argument_error(
                    "RM-CATALOG-BINARY-ARITHMETIC-INPUT",
                    "operands do not form a supported arithmetic pair",
                    0,
                )
            });
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        Err(reason) => output.certainty = runmat_types::CertaintyFact::Dynamic(reason),
    }
    crate::catalog::inference::materialize(&mut output);
    if left.storage == StorageFact::Sparse || right.storage == StorageFact::Sparse {
        let preserves_sparse = matches!(
            policy.sparse_storage,
            SparseStorageRule::PreserveMatchingSparse
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
    if [left, right].iter().any(|fact| {
        matches!(fact.storage, StorageFact::Unknown | StorageFact::Opaque)
            || fact.kind == ValueKindFact::Unknown
    }) {
        output.storage = StorageFact::Unknown;
    }
    if output.storage == StorageFact::Scalar
        && output
            .numeric()
            .is_some_and(|n| n.class == runmat_types::NumericClass::Single)
    {
        output.storage = StorageFact::Dense;
    }
    let materialized_storage = (request.arguments.len() == 2 && diagnostics.is_empty())
        .then(|| materialized_real_storage(left, right, &output.shape))
        .flatten();
    if let Some(storage) = materialized_storage {
        output.storage = storage;
    }
    match output.residency {
        ResidencyFact::Host => {}
        _ if materialized_storage.is_some() => {}
        ResidencyFact::Device { .. } => output.storage = StorageFact::Opaque,
        _ => output.storage = StorageFact::Unknown,
    }
    if matches!(output.storage, StorageFact::Unknown | StorageFact::Opaque) {
        output.layout = LayoutFact::Unknown;
        output.contiguity = ContiguityFact::Unknown;
    }
    if !diagnostics.is_empty() {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::UnsupportedRepresentation);
    } else if let Some(prototype) = admitted.prototype {
        if prototype.certainty != CertaintyFact::Proven {
            output.certainty = prototype.certainty.clone();
        }
    }
    (output, diagnostics)
}

/// Publication carrier is independent of automatic placement and real-result
/// domain admission. Ordinary floating arithmetic reconstructs materialized
/// tensors. Double one-element results may instead scalarize on the host.
fn materialized_real_storage(
    left: &ValueFact,
    right: &ValueFact,
    shape: &ShapeFact,
) -> Option<StorageFact> {
    let numeric = |fact: &ValueFact| {
        let numeric = fact.numeric()?;
        (matches!(fact.storage, StorageFact::Scalar | StorageFact::Dense)
            && numeric.domain == NumericDomain::Real)
            .then_some(numeric.class)
    };
    let class = runmat_types::floating_binary_result_class(numeric(left)?, numeric(right)?)?;
    match (class, shape.element_count()?) {
        (runmat_types::NumericClass::Single, _) => Some(StorageFact::Dense),
        (runmat_types::NumericClass::Double, 1) => None,
        (runmat_types::NumericClass::Double, _) => Some(StorageFact::Dense),
        _ => None,
    }
}

fn dynamic() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}
