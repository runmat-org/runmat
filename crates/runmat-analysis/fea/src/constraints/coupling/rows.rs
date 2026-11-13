//! Original rigid/distributing equations with counted buffers before row expansion.
use super::*;

pub(super) fn rigid_rows(
    coordinates: &impl CoordinateLookup,
    weights: &[ConstraintCouplingWeight],
    reference_point: &ReferencePoint,
    reference_identity: Digest,
    constraint_id: &str,
) -> Result<Vec<CanonicalStructuralConstraintRow>, ConstraintPlanError> {
    let mut rows = super::super::lowering::rows(
        super::super::workspace::multiply(weights.len(), 3)?,
        4,
        constraint_id,
    )?;
    for weight in weights {
        check_cancelled()?;
        let position = coordinates.position(&weight.node_identity).ok_or_else(|| {
            ConstraintPlanError::Invalid(format!(
                "constraint `{constraint_id}` coupling node is absent from solver topology"
            ))
        })?;
        let arm = subtract(position, reference_point.position.canonical_value());
        for component in 0..3 {
            let mut terms = reserve(4)?;
            terms.push(mesh_term(weight.node_identity, component, 1.0));
            terms.push(reference_translation_term(
                reference_identity,
                component,
                -1.0,
            ));
            for rotation in 0..3 {
                let coefficient = -cross(unit_axis(rotation), arm)[component];
                if coefficient != 0.0 {
                    terms.push(reference_rotation_term(
                        reference_identity,
                        rotation,
                        coefficient,
                    ));
                }
            }
            rows.push(row::new_sparse(constraint_id, terms, 0.0)?);
        }
    }
    Ok(rows)
}

pub(super) fn distributing_rows(
    coordinates: &impl CoordinateLookup,
    weights: &[ConstraintCouplingWeight],
    reference_point: &ReferencePoint,
    reference_identity: Digest,
    constraint_id: &str,
) -> Result<Vec<CanonicalStructuralConstraintRow>, ConstraintPlanError> {
    let translation_terms = super::super::workspace::count_add(weights.len(), 4)?;
    let moment_terms = super::super::workspace::count_add(
        super::super::workspace::multiply(weights.len(), 3)?,
        6,
    )?;
    let term_slots = super::super::workspace::multiply(
        super::super::workspace::count_add(translation_terms, moment_terms)?,
        3,
    )?;
    let mut rows =
        super::super::lowering::variable_rows(6, term_slots, moment_terms, constraint_id)?;
    let mut arms = reserve(weights.len())?;
    for weight in weights {
        check_cancelled()?;
        let position = coordinates.position(&weight.node_identity).ok_or_else(|| {
            ConstraintPlanError::Invalid(format!(
                "constraint `{constraint_id}` coupling node is absent from solver topology"
            ))
        })?;
        arms.push(subtract(
            position,
            reference_point.position.canonical_value(),
        ));
    }
    let mean_arm = weights
        .iter()
        .zip(&arms)
        .fold([0.0; 3], |sum, (weight, arm)| {
            super::super::vector::add(sum, scale(*arm, weight.weight))
        });
    let characteristic_length = weights
        .iter()
        .zip(&arms)
        .map(|(weight, arm)| weight.weight * magnitude(*arm).powi(2))
        .sum::<f64>()
        .sqrt();
    if !characteristic_length.is_finite() || characteristic_length <= 1.0e-12 {
        return Err(ConstraintPlanError::Invalid(format!(
            "constraint `{constraint_id}` distributing coupling has no rotational lever arm"
        )));
    }

    for component in 0..3 {
        check_cancelled()?;
        let mut terms = reserve(super::super::workspace::count_add(weights.len(), 4)?)?;
        terms.extend(
            weights
                .iter()
                .map(|weight| mesh_term(weight.node_identity, component, weight.weight)),
        );
        terms.push(reference_translation_term(
            reference_identity,
            component,
            -1.0,
        ));
        for rotation in 0..3 {
            let coefficient = -cross(unit_axis(rotation), mean_arm)[component];
            if coefficient != 0.0 {
                terms.push(reference_rotation_term(
                    reference_identity,
                    rotation,
                    coefficient,
                ));
            }
        }
        rows.push(row::new_sparse(constraint_id, terms, 0.0)?);
    }

    for moment_component in 0..3 {
        check_cancelled()?;
        let mut terms = reserve(super::super::workspace::count_add(
            super::super::workspace::multiply(weights.len(), 3)?,
            6,
        )?)?;
        for (weight, arm) in weights.iter().zip(&arms) {
            for component in 0..3 {
                let coefficient = weight.weight
                    * cross(*arm, unit_axis(component))[moment_component]
                    / characteristic_length;
                if coefficient != 0.0 {
                    terms.push(mesh_term(weight.node_identity, component, coefficient));
                }
            }
        }
        for component in 0..3 {
            let coefficient =
                -cross(mean_arm, unit_axis(component))[moment_component] / characteristic_length;
            if coefficient != 0.0 {
                terms.push(reference_translation_term(
                    reference_identity,
                    component,
                    coefficient,
                ));
            }
        }
        for rotation in 0..3 {
            let coefficient = -weights
                .iter()
                .zip(&arms)
                .map(|(weight, arm)| {
                    weight.weight * cross(*arm, cross(unit_axis(rotation), *arm))[moment_component]
                })
                .sum::<f64>()
                / characteristic_length;
            if coefficient != 0.0 {
                terms.push(reference_rotation_term(
                    reference_identity,
                    rotation,
                    coefficient,
                ));
            }
        }
        rows.push(row::new_sparse(constraint_id, terms, 0.0)?);
    }
    Ok(rows)
}

fn mesh_term(
    identity: StableDigest,
    component: usize,
    coefficient: f64,
) -> CanonicalStructuralConstraintTerm {
    CanonicalStructuralConstraintTerm {
        owner: CanonicalStructuralDofOwner::StructuralNode(identity),
        dof: CanonicalStructuralDof::TRANSLATIONS[component],
        coefficient,
    }
}

fn reference_translation_term(
    identity: Digest,
    component: usize,
    coefficient: f64,
) -> CanonicalStructuralConstraintTerm {
    CanonicalStructuralConstraintTerm {
        owner: CanonicalStructuralDofOwner::ReferencePoint(identity),
        dof: CanonicalStructuralDof::TRANSLATIONS[component],
        coefficient,
    }
}

fn reference_rotation_term(
    identity: Digest,
    component: usize,
    coefficient: f64,
) -> CanonicalStructuralConstraintTerm {
    let dof = match component {
        0 => StructuralRotationDof::Rx,
        1 => StructuralRotationDof::Ry,
        _ => StructuralRotationDof::Rz,
    };
    CanonicalStructuralConstraintTerm {
        owner: CanonicalStructuralDofOwner::ReferencePoint(identity),
        dof: CanonicalStructuralDof::Rotation(dof),
        coefficient,
    }
}

fn unit_axis(component: usize) -> [f64; 3] {
    let mut axis = [0.0; 3];
    axis[component] = 1.0;
    axis
}
