//! Requested construction slots; row/coupling identity encoding remains separate.
use super::super::workspace::{add, count_add, multiply, slots};
use super::*;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintCouplingWorkspaceProfile {
    pub coordinate_slots: usize,
    pub identity_slots: usize,
    pub measure_slots: usize,
    pub weight_slots: usize,
    pub radii_slots: usize,
    pub arm_slots: usize,
    pub row_slots: usize,
    pub retained_term_slots: usize,
    pub compaction_term_slots: usize,
    pub row_id_bytes: u64,
    pub total_requested_bytes: u64,
}
/// Count-only upper bound for original solid coupling. Node inventory bounds weights;
/// explicit surface geometry is separately owned/admitted. Excludes borrowed inputs,
/// canonical identities/row encoding and row stable-sort scratch, allocator rounding/bookkeeping and RSS.
pub fn constraint_coupling_workspace_profile(
    mesh: &SolverMeshArtifact,
    kind: ConstraintCouplingKind,
    id: &str,
) -> Result<ConstraintCouplingWorkspaceProfile, ConstraintPlanError> {
    let mut p = rows_profile(mesh.topology.nodes.len(), kind, id.len())?;
    p.coordinate_slots = mesh.topology.nodes.len();
    p.identity_slots = mesh.topology.nodes.len();
    p.measure_slots = mesh.topology.nodes.len();
    p.total_requested_bytes = add(
        p.total_requested_bytes,
        add(
            slots::<(StableDigest, [f64; 3])>(p.coordinate_slots)?,
            add(
                slots::<(u64, StableDigest)>(p.identity_slots)?,
                slots::<(StableDigest, f64)>(p.measure_slots)?,
            )?,
        )?,
    )?;
    Ok(p)
}
pub(in crate::constraints) fn rows_profile(
    n: usize,
    kind: ConstraintCouplingKind,
    id_bytes: usize,
) -> Result<ConstraintCouplingWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let (rows, terms, compaction, arms) = match kind {
        ConstraintCouplingKind::Rigid => (multiply(n, 3)?, multiply(n, 12)?, 4, 0),
        ConstraintCouplingKind::Distributing => {
            let transl = count_add(n, 4)?;
            let moment = count_add(multiply(n, 3)?, 6)?;
            (6, multiply(count_add(transl, moment)?, 3)?, moment, n)
        }
    };
    let row_id_bytes = u64::try_from(multiply(rows, id_bytes)?)
        .map_err(|_| super::super::workspace::overflow())?;
    let mut total = row_id_bytes;
    for bytes in [
        slots::<ConstraintCouplingWeight>(n)?,
        slots::<f64>(n)?,
        slots::<[f64; 3]>(arms)?,
        slots::<CanonicalStructuralConstraintRow>(rows)?,
        slots::<CanonicalStructuralConstraintTerm>(terms)?,
        slots::<CanonicalStructuralConstraintTerm>(compaction)?,
    ] {
        total = add(total, bytes)?;
    }
    Ok(ConstraintCouplingWorkspaceProfile {
        coordinate_slots: 0,
        identity_slots: 0,
        measure_slots: 0,
        weight_slots: n,
        radii_slots: n,
        arm_slots: arms,
        row_slots: rows,
        retained_term_slots: terms,
        compaction_term_slots: compaction,
        row_id_bytes,
        total_requested_bytes: total,
    })
}
#[cfg(test)]
mod tests;

impl ConstraintCouplingWorkspaceProfile {
    /// Coordinate/weight/lever-arm work excluding row buffers owned by row construction.
    pub(in crate::constraints) fn geometry_workspace_bytes(
        &self,
    ) -> Result<u64, ConstraintPlanError> {
        [
            slots::<(StableDigest, [f64; 3])>(self.coordinate_slots)?,
            slots::<(u64, StableDigest)>(self.identity_slots)?,
            slots::<(StableDigest, f64)>(self.measure_slots)?,
            slots::<ConstraintCouplingWeight>(self.weight_slots)?,
            slots::<f64>(self.radii_slots)?,
            slots::<[f64; 3]>(self.arm_slots)?,
        ]
        .into_iter()
        .try_fold(0, add)
    }
}
