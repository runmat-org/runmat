//! Count retained vector capacities, including unused slots after row/target compaction.
use super::*;
use crate::constraints::workspace::{add, slots};

pub(super) fn row_buffers(
    rows: &[CanonicalStructuralConstraintRow],
    capacity: usize,
) -> Result<u64, ConstraintPlanError> {
    rows.iter().try_fold(
        slots::<CanonicalStructuralConstraintRow>(capacity)?,
        |sum, row| {
            add(
                sum,
                slots::<CanonicalStructuralConstraintTerm>(row.terms.capacity())?,
            )
        },
    )
}

pub(super) fn evidence_buffers(
    targets: &Vec<ResolvedConstraintTarget>,
    pairings: &Vec<ConstraintPairingEvidence>,
    coupling: Option<&ConstraintCouplingEvidence>,
) -> Result<u64, ConstraintPlanError> {
    let mut bytes = add(
        slots::<ResolvedConstraintTarget>(targets.capacity())?,
        slots::<ConstraintPairingEvidence>(pairings.capacity())?,
    )?;
    for target in targets {
        bytes = add(
            bytes,
            slots::<runmat_meshing_core::PersistentEntityId>(target.matched_entities.capacity())?,
        )?;
        bytes = add(
            bytes,
            slots::<runmat_meshing_core::StableDigest>(target.matched_node_identities.capacity())?,
        )?;
    }
    for pairing in pairings {
        bytes = add(
            bytes,
            slots::<ConstraintPairingWeight>(pairing.primary_weights.capacity())?,
        )?;
    }
    if let Some(coupling) = coupling {
        bytes = add(
            bytes,
            slots::<ConstraintCouplingWeight>(coupling.target_weights.capacity())?,
        )?;
    }
    Ok(bytes)
}

pub(super) fn lowered_bytes(compiled: &CompiledConstraint) -> Result<u64, ConstraintPlanError> {
    let payload = super::super::plan::canonical_value_workspace_bytes(&(
        &compiled.targets,
        &compiled.rows,
        &compiled.pairings,
        &compiled.coupling,
    ))?;
    add(
        payload,
        add(
            row_buffers(&compiled.rows, compiled.rows.capacity())?,
            evidence_buffers(
                &compiled.targets,
                &compiled.pairings,
                compiled.coupling.as_ref(),
            )?,
        )?,
    )
}
