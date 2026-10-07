//! Additional row finalization storage, excluding the incoming term allocation.
use super::*;
use crate::constraints::workspace::{add, check_cancelled, ordering_workspace_bytes, slots};

/// Requested additional slots plus the shared canonical materialization estimate.
/// The borrowed incoming term buffer/capacity is already owned by lowering. Compaction
/// reserves a possible replacement buffer; equal-term summation can only shrink it.
/// Excludes allocator bookkeeping/rounding, borrowed terms, error strings and RSS.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConstraintRowWorkspaceProfile {
    pub ordering_bytes: u64,
    pub compaction_bytes: u64,
    pub id_copy_bytes: u64,
    pub identity_materialization_bytes: u64,
    pub total_requested_bytes: u64,
}
pub fn constraint_row_workspace_profile(
    id: &str,
    terms: &[CanonicalStructuralConstraintTerm],
    rhs: f64,
) -> Result<ConstraintRowWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let identity_materialization_bytes = super::super::plan::canonical_value_workspace_bytes(&(
        id,
        terms,
        rhs,
        ConstraintResidualQuantity::Displacement,
        ReactionRecovery::UnconstrainedEquilibriumResidual,
    ))?;
    derive(id, terms.len(), identity_materialization_bytes)
}

/// Count-only identity shape before incoming terms exist. Inventory values only describe
/// maximal serialized widths; they never become row values, identities or physics evidence.
pub(in crate::constraints) fn from_term_count(
    id: &str,
    count: usize,
) -> Result<ConstraintRowWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    slots::<CanonicalStructuralConstraintTerm>(count)?;
    struct Terms(usize);
    impl serde::Serialize for Terms {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeSeq;
            let term = CanonicalStructuralConstraintTerm {
                owner: CanonicalStructuralDofOwner::StructuralNode(StableDigest::from_bytes(
                    [255; 32],
                )),
                dof: CanonicalStructuralDof::TRANSLATIONS[0],
                coefficient: f64::MAX,
            };
            let mut sequence = serializer.serialize_seq(Some(self.0))?;
            for ordinal in 0..self.0 {
                if ordinal % 1024 == 0 && crate::is_fea_cancelled() {
                    return Err(serde::ser::Error::custom(
                        "constraint row profiling cancelled",
                    ));
                }
                sequence.serialize_element(&term)?;
            }
            sequence.end()
        }
    }
    let identity_materialization_bytes = super::super::plan::canonical_value_workspace_bytes(&(
        id,
        Terms(count),
        f64::MAX,
        ConstraintResidualQuantity::Displacement,
        ReactionRecovery::UnconstrainedEquilibriumResidual,
    ))?;
    derive(id, count, identity_materialization_bytes)
}

fn derive(
    id: &str,
    count: usize,
    identity_materialization_bytes: u64,
) -> Result<ConstraintRowWorkspaceProfile, ConstraintPlanError> {
    let ordering_bytes = ordering_workspace_bytes(count)?;
    let compaction_bytes = slots::<CanonicalStructuralConstraintTerm>(count)?;
    let id_copy_bytes = id.len() as u64;
    let total_requested_bytes = add(
        add(ordering_bytes, compaction_bytes)?,
        add(id_copy_bytes, identity_materialization_bytes)?,
    )?;
    Ok(ConstraintRowWorkspaceProfile {
        ordering_bytes,
        compaction_bytes,
        id_copy_bytes,
        identity_materialization_bytes,
        total_requested_bytes,
    })
}
