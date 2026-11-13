//! Borrowed reference-point preparation; clone only unique authored IDs.
use super::super::workspace::{add, slots};
use super::*;

/// Counted reference/record/ID-copy slots and the largest shared identity materialization,
/// excluding borrowed model, allocator rounding/bookkeeping and RSS. Repeated IDs are bounded before deduplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintReferencePointWorkspaceProfile {
    pub candidate_slots: usize,
    pub point_slots: usize,
    pub id_payload_bytes: u64,
    pub identity_materialization_bytes: u64,
    pub total_requested_bytes: u64,
}
pub fn constraint_reference_point_workspace_profile(
    model: &AnalysisModel,
) -> Result<ConstraintReferencePointWorkspaceProfile, ConstraintPlanError> {
    from_constraints(
        model
            .constraints
            .iter()
            .filter_map(|constraint| constraint.structural()),
    )
}

fn from_constraints<'a>(
    constraints: impl IntoIterator<Item = &'a StructuralConstraint>,
) -> Result<ConstraintReferencePointWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let mut identity_materialization_bytes = 0;
    let mut count = 0usize;
    let mut bytes = 0u64;
    for constraint in constraints {
        check_cancelled()?;
        let coupling = match &constraint.kind {
            StructuralConstraintKind::RigidCoupling(c)
            | StructuralConstraintKind::DistributingCoupling(c) => c,
            _ => continue,
        };
        count = super::super::workspace::count_add(count, 1)?;
        identity_materialization_bytes = identity_materialization_bytes.max(
            super::super::plan::canonical_value_workspace_bytes(&(
                coupling.reference_point.reference_point_id.trim(),
                coupling.reference_point.position.canonical_value(),
            ))?,
        );
        bytes = add(
            bytes,
            coupling.reference_point.reference_point_id.trim().len() as u64,
        )?;
    }
    let total = add(
        identity_materialization_bytes,
        add(
            bytes,
            add(
                slots::<(&str, [f64; 3], usize)>(count)?,
                slots::<CanonicalStructuralReferencePoint>(count)?,
            )?,
        )?,
    )?;
    usize::try_from(total).map_err(|_| super::super::workspace::overflow())?;
    Ok(ConstraintReferencePointWorkspaceProfile {
        candidate_slots: count,
        point_slots: count,
        id_payload_bytes: bytes,
        identity_materialization_bytes,
        total_requested_bytes: total,
    })
}
pub(in crate::constraints) fn canonical_reference_points(
    constraints: &[&StructuralConstraint],
) -> Result<Vec<CanonicalStructuralReferencePoint>, ConstraintPlanError> {
    let profile = from_constraints(constraints.iter().copied())?;
    let count = profile.candidate_slots;
    let mut by_id = reserve(count)?;
    for constraint in constraints {
        check_cancelled()?;
        let coupling = match &constraint.kind {
            StructuralConstraintKind::RigidCoupling(c)
            | StructuralConstraintKind::DistributingCoupling(c) => c,
            _ => continue,
        };
        let id = coupling.reference_point.reference_point_id.trim();
        let position = coupling.reference_point.position.canonical_value();
        if id.is_empty() || position.iter().any(|v| !v.is_finite()) {
            return Err(ConstraintPlanError::Invalid(format!(
                "constraint `{}` has an invalid reference point",
                constraint.constraint_id
            )));
        }
        by_id.push((id, position, by_id.len()));
    }
    // Authored ordinal makes equal IDs deterministic without stable-sort scratch.
    by_id.sort_unstable_by_key(|(id, _, ordinal)| (*id, *ordinal));
    for pair in by_id.windows(2) {
        if pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1 {
            return Err(ConstraintPlanError::Invalid(format!(
                "reference point `{}` has conflicting positions",
                pair[0].0
            )));
        }
    }
    // The original map overwrote the value on an equal ID, including signed zero.
    by_id.dedup_by(|later, earlier| {
        if later.0 == earlier.0 {
            *earlier = *later;
            true
        } else {
            false
        }
    });
    let mut points = reserve(by_id.len())?;
    for (id, position, _) in by_id {
        check_cancelled()?;
        points.push(CanonicalStructuralReferencePoint {
            identity: reference_point_identity(id, position)?,
            reference_point_id: id.into(),
            position_meters: position,
        });
    }
    points.sort_unstable_by_key(|p| p.identity);
    if points
        .windows(2)
        .any(|pair| pair[0].identity == pair[1].identity)
    {
        return Err(ConstraintPlanError::Invalid(
            "reference-point identities collide".into(),
        ));
    }
    Ok(points)
}
