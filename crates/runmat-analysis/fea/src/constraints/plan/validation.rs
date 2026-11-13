//! Canonical validation borrows already-sorted points and rows; no copied lookup sets.
use super::*;
pub(super) fn validate(
    plan: &CanonicalStructuralConstraintPlan,
) -> Result<(), ConstraintPlanError> {
    super::super::workspace::check_cancelled()?;
    if plan.schema_version != 1
        || !plan.topology.has_valid_identity()
        || plan.evidence.authored_constraint_count != plan.groups.len()
        || plan
            .evidence
            .active_row_count
            .checked_add(plan.evidence.redundant_row_count)
            != Some(plan.rows.len())
        || plan.evidence.rank != plan.evidence.active_row_count
        || plan.evidence.conflicting_row_count != 0
    {
        return Err(ConstraintPlanError::Invalid(
            "schema, topology, or evidence is inconsistent".into(),
        ));
    }
    if !plan
        .reference_points
        .windows(2)
        .all(|pair| pair[0].identity < pair[1].identity)
        || plan.reference_points.iter().any(|point| {
            point.reference_point_id.trim().is_empty()
                || point.position_meters.iter().any(|value| !value.is_finite())
                || match reference_point_identity(&point.reference_point_id, point.position_meters)
                {
                    Ok(identity) => identity != point.identity,
                    Err(_) => true,
                }
        })
    {
        return Err(ConstraintPlanError::Invalid(
            "canonical reference-point evidence is invalid".into(),
        ));
    }
    if !enforcement::matches_rows(plan.enforcement, &plan.rows) {
        return Err(ConstraintPlanError::Invalid(
            "plan enforcement differs from its canonical row space".into(),
        ));
    }
    if !plan
        .groups
        .windows(2)
        .all(|pair| pair[0].constraint_id < pair[1].constraint_id)
        || !plan
            .rows
            .windows(2)
            .all(|pair| compare_rows(&pair[0], &pair[1]) == Ordering::Less)
    {
        return Err(ConstraintPlanError::Invalid(
            "groups and rows must be uniquely ordered".into(),
        ));
    }
    let mut grouped_rows = 0;
    for group in &plan.groups {
        super::super::workspace::check_cancelled()?;
        // compare_rows orders constraint IDs first, and strict ordering was checked above.
        let start = plan
            .rows
            .partition_point(|row| row.constraint_id < group.constraint_id);
        let end = plan
            .rows
            .partition_point(|row| row.constraint_id <= group.constraint_id);
        let group_row_count = end - start;
        grouped_rows = super::super::workspace::count_add(grouped_rows, group_row_count)?;
        let implementation_is_valid = match plan.topology {
                CanonicalStructuralTopologyIdentity::Solid { .. } => {
                    group.implementation
                        == runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection
                }
                CanonicalStructuralTopologyIdentity::ConformalMixed { .. } => matches!(
                    group.implementation,
                    runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection
                        | runmat_analysis_core::BoundaryTransferImplementation::ExplicitRegionConnectivity
                ),
                CanonicalStructuralTopologyIdentity::Explicit { .. } => {
                    group.implementation
                        == runmat_analysis_core::BoundaryTransferImplementation::ExplicitRegionConnectivity
                }
            };
        if group.constraint_id.is_empty()
            || !implementation_is_valid
            || group.targets.is_empty()
            || group.row_count != group_row_count
            || !group
                .targets
                .windows(2)
                .all(|pair| pair[0].selection_digest < pair[1].selection_digest)
            || !group
                .reference_point_identities
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            || group
                .reference_point_identities
                .iter()
                .any(|identity| !contains_reference(plan, identity))
            || match &group.coupling {
                Some(coupling) => {
                    group.reference_point_identities.as_slice()
                        != [coupling.reference_point_identity]
                        || coupling.target_weights.is_empty()
                        || !coupling.characteristic_length_meters.is_finite()
                        || coupling.characteristic_length_meters < 0.0
                        || (coupling.kind == ConstraintCouplingKind::Distributing
                            && coupling.characteristic_length_meters <= 1.0e-12)
                        || !coupling
                            .target_weights
                            .windows(2)
                            .all(|pair| pair[0].node_identity < pair[1].node_identity)
                        || coupling.target_weights.iter().any(|weight| {
                            weight.node_identity == StableDigest::ZERO
                                || !weight.weight.is_finite()
                                || weight.weight <= 0.0
                                || !group.targets.iter().any(|target| {
                                    target
                                        .matched_node_identities
                                        .contains(&weight.node_identity)
                                })
                        })
                        || (coupling
                            .target_weights
                            .iter()
                            .map(|weight| weight.weight)
                            .sum::<f64>()
                            - 1.0)
                            .abs()
                            > 1.0e-10
                }
                None => !group.reference_point_identities.is_empty(),
            }
            || !group
                .pairings
                .windows(2)
                .all(|pair| pair[0].secondary_node_identity < pair[1].secondary_node_identity)
            || group.pairings.iter().any(|pairing| {
                pairing.secondary_node_identity == StableDigest::ZERO
                    || !pairing.geometric_error_meters.is_finite()
                    || pairing.geometric_error_meters < 0.0
                    || pairing.primary_weights.is_empty()
                    || !pairing
                        .primary_weights
                        .windows(2)
                        .all(|pair| pair[0].primary_node_identity < pair[1].primary_node_identity)
                    || pairing.primary_weights.iter().any(|weight| {
                        weight.primary_node_identity == StableDigest::ZERO
                            || !weight.weight.is_finite()
                            || weight.weight == 0.0
                    })
                    || (pairing
                        .primary_weights
                        .iter()
                        .map(|weight| weight.weight)
                        .sum::<f64>()
                        - 1.0)
                        .abs()
                        > 1.0e-10
            })
        {
            return Err(ConstraintPlanError::Invalid(
                "group evidence differs from its canonical rows".into(),
            ));
        }
        for target in &group.targets {
            validate_resolved_target(&group.constraint_id, target)?;
        }
    }
    if grouped_rows != plan.rows.len() {
        return Err(ConstraintPlanError::Invalid(
            "constraint rows include an undeclared group".into(),
        ));
    }
    for row in &plan.rows {
        super::super::workspace::check_cancelled()?;
        if row.constraint_id.is_empty()
            || row.terms.is_empty()
            || !row.right_hand_side.is_finite()
            || row.identity != row_identity(row)?
            || row.residual_quantity != residual_quantity(&row.terms)
        {
            return Err(ConstraintPlanError::Invalid(
                "constraint row identity or numerical content is invalid".into(),
            ));
        }
        if row.terms.iter().any(|term| {
            let invalid_owner = match term.owner {
                CanonicalStructuralDofOwner::StructuralNode(identity) => {
                    identity == StableDigest::ZERO
                }
                CanonicalStructuralDofOwner::ReferencePoint(identity) => {
                    !contains_reference(plan, &identity)
                }
            };
            invalid_owner || !term.coefficient.is_finite() || term.coefficient == 0.0
        }) || !row
            .terms
            .windows(2)
            .all(|pair| term_key(&pair[0]) < term_key(&pair[1]))
        {
            return Err(ConstraintPlanError::Invalid(
                "constraint row terms are invalid or not uniquely ordered".into(),
            ));
        }
    }
    let (active, redundant) = validate_row_classification(&plan.rows)?;
    if active != plan.evidence.active_row_count || redundant != plan.evidence.redundant_row_count {
        return Err(ConstraintPlanError::Invalid(
            "constraint rank evidence differs from its canonical rows".into(),
        ));
    }
    Ok(())
}
fn contains_reference(plan: &CanonicalStructuralConstraintPlan, id: &Digest) -> bool {
    plan.reference_points
        .binary_search_by_key(id, |point| point.identity)
        .is_ok()
}
