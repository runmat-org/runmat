//! Borrowed authored-source inventory before compilation's source/ID buffers.
use super::*;
use crate::constraints::workspace::{add, check_cancelled, ordering_workspace_bytes, slots};

/// Requested vectors and shared canonical materialization estimate for source preparation.
/// Equation targets are borrowed; no copied strings/entity metadata are included. Excludes
/// borrowed model, reference-point preparation, lowering/topology, allocator overhead/RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintAuthoredWorkspaceProfile {
    pub constraint_count: usize,
    pub borrowed_source_bytes: u64,
    pub unique_id_bytes: u64,
    pub source_digest_bytes: u64,
    pub equation_reference_bytes: u64,
    pub equation_ordering_bytes: u64,
    pub canonical_materialization_bytes: u64,
    pub total_requested_bytes: u64,
}
pub fn constraint_authored_workspace_profile(
    model: &AnalysisModel,
) -> Result<ConstraintAuthoredWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let mut count = 0;
    let mut equation_reference_bytes = 0;
    let mut equation_ordering_bytes = 0;
    let mut canonical_materialization_bytes = 0;
    for constraint in &model.constraints {
        check_cancelled()?;
        let Some(c) = constraint.structural() else {
            continue;
        };
        count = crate::constraints::workspace::count_add(count, 1)?;
        let source = constraint_source_workspace_profile(c)?;
        equation_reference_bytes = equation_reference_bytes.max(source.equation_reference_bytes);
        equation_ordering_bytes = equation_ordering_bytes.max(source.equation_ordering_bytes);
        canonical_materialization_bytes =
            canonical_materialization_bytes.max(source.canonical_materialization_bytes);
    }
    let borrowed_source_bytes = slots::<&runmat_analysis_core::StructuralConstraint>(count)?;
    let unique_id_bytes = slots::<(&str, usize)>(count)?;
    let source_digest_bytes = slots::<Digest>(count)?;
    // Sorted source digest collection uses its own contract envelope; count without allocating it.
    #[derive(Serialize)]
    struct DigestInventory(#[serde(serialize_with = "digest_inventory")] usize);
    fn digest_inventory<S: serde::Serializer>(
        count: &usize,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let digest = Digest::from_bytes([0; 32]);
        let mut seq = serializer.serialize_seq(Some(*count))?;
        for ordinal in 0..*count {
            if ordinal % 1024 == 0 && crate::is_fea_cancelled() {
                return Err(serde::ser::Error::custom("constraint profiling cancelled"));
            }
            seq.serialize_element(&digest)?;
        }
        seq.end()
    }
    canonical_materialization_bytes = canonical_materialization_bytes
        .max(canonical_value_workspace_bytes(&DigestInventory(count))?);
    let mut total_requested_bytes = canonical_materialization_bytes;
    for bytes in [
        borrowed_source_bytes,
        unique_id_bytes,
        source_digest_bytes,
        equation_reference_bytes,
        equation_ordering_bytes,
    ] {
        total_requested_bytes = add(total_requested_bytes, bytes)?;
    }
    Ok(ConstraintAuthoredWorkspaceProfile {
        constraint_count: count,
        borrowed_source_bytes,
        unique_id_bytes,
        source_digest_bytes,
        equation_reference_bytes,
        equation_ordering_bytes,
        canonical_materialization_bytes,
        total_requested_bytes,
    })
}

/// Additional source-digest construction, excluding its borrowed authored constraint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintSourceWorkspaceProfile {
    pub equation_reference_bytes: u64,
    pub equation_ordering_bytes: u64,
    pub canonical_materialization_bytes: u64,
    pub total_requested_bytes: u64,
}

pub fn constraint_source_workspace_profile(
    constraint: &runmat_analysis_core::StructuralConstraint,
) -> Result<ConstraintSourceWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let equation_count = match &constraint.kind {
        StructuralConstraintKind::Equation(equation) => equation.terms.len(),
        _ => 0,
    };
    let equation_reference_bytes =
        slots::<&runmat_analysis_core::StructuralConstraintEquationTerm>(equation_count)?;
    let equation_ordering_bytes = ordering_workspace_bytes(equation_count)?;
    // The authored struct bounds its canonical source tuple, including equation projection.
    let canonical_materialization_bytes = canonical_value_workspace_bytes(constraint)?;
    let total_requested_bytes = add(
        add(equation_reference_bytes, equation_ordering_bytes)?,
        canonical_materialization_bytes,
    )?;
    Ok(ConstraintSourceWorkspaceProfile {
        equation_reference_bytes,
        equation_ordering_bytes,
        canonical_materialization_bytes,
        total_requested_bytes,
    })
}
