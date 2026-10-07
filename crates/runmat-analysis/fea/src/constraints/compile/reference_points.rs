//! Borrowed reference-point preparation; clone only unique authored IDs.
use super::super::workspace::{add, slots};
use super::*;

/// Counted reference/record/ID-copy slots, excluding borrowed model, identity encoding,
/// allocator rounding/bookkeeping and RSS. Repeated IDs are bounded before deduplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintReferencePointWorkspaceProfile {
    pub candidate_slots: usize,
    pub point_slots: usize,
    pub id_payload_bytes: u64,
    pub total_requested_bytes: u64,
}
pub fn constraint_reference_point_workspace_profile(
    model: &AnalysisModel,
) -> Result<ConstraintReferencePointWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let mut count = 0usize;
    let mut bytes = 0u64;
    for constraint in &model.constraints {
        check_cancelled()?;
        let runmat_analysis_core::AnalysisConstraint::Structural(c) = constraint else {
            continue;
        };
        let coupling = match &c.kind {
            StructuralConstraintKind::RigidCoupling(c)
            | StructuralConstraintKind::DistributingCoupling(c) => c,
            _ => continue,
        };
        count = super::super::workspace::count_add(count, 1)?;
        bytes = add(
            bytes,
            coupling.reference_point.reference_point_id.trim().len() as u64,
        )?;
    }
    let total = add(
        bytes,
        add(
            slots::<(&str, [f64; 3], usize)>(count)?,
            slots::<CanonicalStructuralReferencePoint>(count)?,
        )?,
    )?;
    Ok(ConstraintReferencePointWorkspaceProfile {
        candidate_slots: count,
        point_slots: count,
        id_payload_bytes: bytes,
        total_requested_bytes: total,
    })
}
pub(in crate::constraints) fn canonical_reference_points(
    constraints: &[&StructuralConstraint],
) -> Result<Vec<CanonicalStructuralReferencePoint>, ConstraintPlanError> {
    let count = constraints
        .iter()
        .filter(|c| {
            matches!(
                c.kind,
                StructuralConstraintKind::RigidCoupling(_)
                    | StructuralConstraintKind::DistributingCoupling(_)
            )
        })
        .count();
    let mut by_id = reserve(count)?;
    for constraint in constraints {
        check_cancelled()?;
        let coupling = match &constraint.kind {
            StructuralConstraintKind::RigidCoupling(c)
            | StructuralConstraintKind::DistributingCoupling(c) => c,
            _ => continue,
        };
        let id = coupling.reference_point.reference_point_id.trim();
        let position = coupling.reference_point.position.meters;
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
