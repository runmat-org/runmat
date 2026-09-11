use std::collections::BTreeSet;

use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, PrescribedStructuralDofs, StructuralConstraint,
    StructuralConstraintKind, StructuralRotationDof, TargetCardinality,
};
use runmat_meshing_core::SolverMeshArtifact;

use crate::physics::structural::topology::{
    explicit_target_node_indices,
    node_space::CanonicalStructuralNodeSpace,
    target::{resolve_mixed_structural_target, MixedStructuralTargetOwnership},
};

use super::{
    contract::{
        CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintRow,
        CanonicalStructuralConstraintTerm, CanonicalStructuralDof, CanonicalStructuralDofOwner,
        ConstraintGroupEvidence, ConstraintPlanError, ConstraintPlanEvidence,
        ResolvedConstraintTarget,
    },
    enforcement,
    frame::{require_global, translation_basis, translation_direction},
    plan::{authored_constraints_digest, source_digest, structural_constraints},
    row,
    target::resolve_solid_target,
};

pub(super) fn compile(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<CanonicalStructuralConstraintPlan, ConstraintPlanError> {
    let structural = model.structural.as_ref().ok_or_else(|| {
        ConstraintPlanError::Invalid(
            "mixed constraint compilation requires explicit topology".into(),
        )
    })?;
    let topology = CanonicalStructuralNodeSpace::derive(mesh, structural)
        .map_err(|error| ConstraintPlanError::Invalid(error.to_string()))?;
    let authored = structural_constraints(model);
    let mut solid_model = model.clone();
    solid_model.constraints.clear();
    let mut generic = Vec::new();
    for constraint in &authored {
        if requires_mixed_compilation(constraint, mesh, structural, &topology)? {
            generic.push(*constraint);
        } else {
            let authored = model
                .constraints
                .iter()
                .find(|candidate| {
                    candidate.structural().is_some_and(|candidate| {
                        candidate.constraint_id == constraint.constraint_id
                    })
                })
                .expect("structural constraint came from the authored model")
                .clone();
            solid_model.constraints.push(authored);
        }
    }
    let mut plan = super::compile::compile(&solid_model, mesh)?;
    let mut enforcement_mode = plan.enforcement;
    for constraint in generic {
        let (mut targets, rows) =
            compile_node_constraint(model, Some(mesh), structural, &topology, constraint)?;
        targets.sort_by_key(|target| target.selection_digest);
        targets.dedup_by_key(|target| target.selection_digest);
        enforcement_mode = enforcement::combine(
            enforcement_mode,
            enforcement::select(&constraint.kind, &rows),
        );
        plan.groups.push(ConstraintGroupEvidence {
            constraint_id: constraint.constraint_id.clone(),
            source_digest: source_digest(constraint)?,
            implementation:
                runmat_analysis_core::BoundaryTransferImplementation::ExplicitRegionConnectivity,
            targets,
            reference_point_identities: Vec::new(),
            coupling: None,
            pairings: Vec::new(),
            row_count: rows.len(),
        });
        plan.rows.extend(rows);
    }
    plan.groups
        .sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
    plan.rows.sort_by(row::compare_rows);
    let (active_row_count, redundant_row_count) = row::classify(&mut plan.rows)?;
    plan.topology = topology.identity;
    plan.authored_constraints_digest = authored_constraints_digest(&authored)?;
    plan.enforcement = enforcement_mode;
    plan.evidence = ConstraintPlanEvidence {
        authored_constraint_count: plan.groups.len(),
        active_row_count,
        redundant_row_count,
        conflicting_row_count: 0,
        rank: active_row_count,
    };
    plan.validate()?;
    Ok(plan)
}

pub(super) fn compile_explicit(
    model: &AnalysisModel,
) -> Result<CanonicalStructuralConstraintPlan, ConstraintPlanError> {
    let structural = model.structural.as_ref().ok_or_else(|| {
        ConstraintPlanError::Invalid(
            "explicit constraint compilation requires structural topology".into(),
        )
    })?;
    let topology = CanonicalStructuralNodeSpace::derive_explicit(structural)
        .map_err(|error| ConstraintPlanError::Invalid(error.to_string()))?;
    let authored = structural_constraints(model);
    let mut constraint_ids = BTreeSet::new();
    for constraint in &authored {
        if !constraint_ids.insert(constraint.constraint_id.as_str()) {
            return Err(ConstraintPlanError::DuplicateConstraintId {
                constraint_id: constraint.constraint_id.clone(),
            });
        }
    }
    let mut groups = Vec::with_capacity(authored.len());
    let mut rows = Vec::new();
    let mut enforcement_mode = super::contract::ConstraintEnforcement::DirectElimination;
    for constraint in &authored {
        let (mut targets, compiled_rows) =
            compile_node_constraint(model, None, structural, &topology, constraint)?;
        targets.sort_by_key(|target| target.selection_digest);
        targets.dedup_by_key(|target| target.selection_digest);
        enforcement_mode = enforcement::combine(
            enforcement_mode,
            enforcement::select(&constraint.kind, &compiled_rows),
        );
        groups.push(ConstraintGroupEvidence {
            constraint_id: constraint.constraint_id.clone(),
            source_digest: source_digest(constraint)?,
            implementation:
                runmat_analysis_core::BoundaryTransferImplementation::ExplicitRegionConnectivity,
            targets,
            reference_point_identities: Vec::new(),
            coupling: None,
            pairings: Vec::new(),
            row_count: compiled_rows.len(),
        });
        rows.extend(compiled_rows);
    }
    groups.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
    rows.sort_by(row::compare_rows);
    let (active_row_count, redundant_row_count) = row::classify(&mut rows)?;
    let plan = CanonicalStructuralConstraintPlan {
        schema_version: 1,
        topology: topology.identity,
        authored_constraints_digest: authored_constraints_digest(&authored)?,
        enforcement: enforcement_mode,
        reference_points: Vec::new(),
        evidence: ConstraintPlanEvidence {
            authored_constraint_count: groups.len(),
            active_row_count,
            redundant_row_count,
            conflicting_row_count: 0,
            rank: active_row_count,
        },
        groups,
        rows,
    };
    plan.validate()?;
    Ok(plan)
}

fn requires_mixed_compilation(
    constraint: &StructuralConstraint,
    mesh: &SolverMeshArtifact,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
) -> Result<bool, ConstraintPlanError> {
    let mut mixed = false;
    visit_targets(&constraint.kind, |target| {
        if mixed {
            return;
        }
        mixed = !matches!(
            resolve_mixed_structural_target(&target.region_id, mesh, structural, topology),
            Ok(MixedStructuralTargetOwnership::Solid { .. })
        );
    });
    if matches!(constraint.kind, StructuralConstraintKind::Equation(_)) {
        require_global(constraint)?;
    }
    Ok(mixed)
}

fn compile_node_constraint(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
) -> Result<
    (
        Vec<ResolvedConstraintTarget>,
        Vec<CanonicalStructuralConstraintRow>,
    ),
    ConstraintPlanError,
> {
    match &constraint.kind {
        StructuralConstraintKind::Fixed { target } => {
            require_global(constraint)?;
            let (resolved, nodes) = resolve_target(mesh, structural, topology, constraint, target)?;
            let mut rows = Vec::new();
            for node in nodes {
                for dof in available_dofs(topology, node) {
                    rows.push(row::new_singleton(
                        &constraint.constraint_id,
                        topology.nodes[node].identity,
                        dof,
                        1.0,
                        0.0,
                    )?);
                }
            }
            Ok((vec![resolved], rows))
        }
        StructuralConstraintKind::Prescribed(values) => {
            compile_prescribed(model, mesh, structural, topology, constraint, values)
        }
        StructuralConstraintKind::Equation(equation) => {
            require_global(constraint)?;
            if equation.terms.is_empty() {
                return Err(ConstraintPlanError::Invalid(format!(
                    "constraint `{}` has no equation terms",
                    constraint.constraint_id
                )));
            }
            let mut targets = Vec::new();
            let mut terms = Vec::new();
            for term in &equation.terms {
                if !term.coefficient.is_finite() || term.coefficient == 0.0 {
                    return Err(ConstraintPlanError::Invalid(format!(
                        "constraint `{}` contains a zero or non-finite equation coefficient",
                        constraint.constraint_id
                    )));
                }
                let (resolved, nodes) =
                    resolve_target(mesh, structural, topology, constraint, &term.target)?;
                for node in nodes {
                    let dof = CanonicalStructuralDof::from(term.dof);
                    if !available_dofs(topology, node).contains(&dof) {
                        return Err(ConstraintPlanError::UnavailableRotationalDof {
                            constraint_id: constraint.constraint_id.clone(),
                        });
                    }
                    terms.push(CanonicalStructuralConstraintTerm {
                        owner: CanonicalStructuralDofOwner::StructuralNode(
                            topology.nodes[node].identity,
                        ),
                        dof,
                        coefficient: term.coefficient,
                    });
                }
                targets.push(resolved);
            }
            if !equation.right_hand_side.meters.is_finite() {
                return Err(ConstraintPlanError::NonFiniteValue {
                    constraint_id: constraint.constraint_id.clone(),
                });
            }
            Ok((
                targets,
                vec![row::new_sparse(
                    &constraint.constraint_id,
                    terms,
                    equation.right_hand_side.meters,
                )?],
            ))
        }
        _ => Err(ConstraintPlanError::UnsupportedTarget {
            constraint_id: constraint.constraint_id.clone(),
            reason: "this relation is not implemented on explicit structural topology",
        }),
    }
}

fn compile_prescribed(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    values: &PrescribedStructuralDofs,
) -> Result<
    (
        Vec<ResolvedConstraintTarget>,
        Vec<CanonicalStructuralConstraintRow>,
    ),
    ConstraintPlanError,
> {
    let (resolved, nodes) = resolve_target(mesh, structural, topology, constraint, &values.target)?;
    let mut rows = Vec::new();
    for node in nodes {
        let canonical = &topology.nodes[node];
        let basis = translation_basis(
            &constraint.frame,
            canonical.coordinates_m,
            None,
            &constraint.constraint_id,
        )?;
        for prescribed in &values.translations {
            let value = prescribed
                .value
                .study_constant(&model.signals)
                .ok_or_else(|| ConstraintPlanError::ScheduledValue {
                    constraint_id: constraint.constraint_id.clone(),
                })?
                .meters;
            rows.push(row::new_direction(
                &constraint.constraint_id,
                canonical.identity,
                translation_direction(prescribed.dof.into(), basis),
                finite_value(value, constraint)?,
            )?);
        }
        if !values.rotations.is_empty() && !canonical.owns_explicit_rotation {
            return Err(ConstraintPlanError::UnavailableRotationalDof {
                constraint_id: constraint.constraint_id.clone(),
            });
        }
        if !values.rotations.is_empty() && constraint.frame != CoordinateFrame::Global {
            return Err(ConstraintPlanError::UnsupportedFrame {
                constraint_id: constraint.constraint_id.clone(),
                frame: format!("{:?}", constraint.frame),
            });
        }
        for prescribed in &values.rotations {
            let value = prescribed
                .value
                .study_constant(&model.signals)
                .ok_or_else(|| ConstraintPlanError::ScheduledValue {
                    constraint_id: constraint.constraint_id.clone(),
                })?
                .radians;
            rows.push(row::new_singleton(
                &constraint.constraint_id,
                canonical.identity,
                CanonicalStructuralDof::Rotation(prescribed.dof),
                1.0,
                finite_value(value, constraint)?,
            )?);
        }
    }
    Ok((vec![resolved], rows))
}

fn finite_value(value: f64, constraint: &StructuralConstraint) -> Result<f64, ConstraintPlanError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| ConstraintPlanError::NonFiniteValue {
            constraint_id: constraint.constraint_id.clone(),
        })
}

fn resolve_target(
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    target: &runmat_analysis_core::RegionTarget,
) -> Result<(ResolvedConstraintTarget, Vec<usize>), ConstraintPlanError> {
    let ownership = if let Some(mesh) = mesh {
        resolve_mixed_structural_target(&target.region_id, mesh, structural, topology).map_err(
            |error| {
                ConstraintPlanError::Invalid(format!(
                    "constraint `{}` target resolution failed: {error}",
                    constraint.constraint_id
                ))
            },
        )?
    } else {
        let canonical_nodes = explicit_target_node_indices(structural, &target.region_id)
            .into_iter()
            .map(|node| topology.explicit_node_indices[node])
            .collect::<Vec<_>>();
        if canonical_nodes.is_empty() {
            return Err(ConstraintPlanError::EmptyTarget {
                constraint_id: constraint.constraint_id.clone(),
                region_id: target.region_id.clone(),
            });
        }
        MixedStructuralTargetOwnership::Explicit { canonical_nodes }
    };
    let nodes = ownership.canonical_nodes().to_vec();
    let resolved = match ownership {
        MixedStructuralTargetOwnership::Solid { .. }
        | MixedStructuralTargetOwnership::Shared { .. } => resolve_solid_target(
            mesh.expect("solid target ownership requires a solver mesh"),
            target,
            &constraint.constraint_id,
        )?,
        MixedStructuralTargetOwnership::Explicit { .. } => {
            let mut identities = nodes
                .iter()
                .map(|node| topology.nodes[*node].identity)
                .collect::<Vec<_>>();
            identities.sort_unstable();
            identities.dedup();
            if target.cardinality == TargetCardinality::One && identities.len() != 1 {
                return Err(ConstraintPlanError::Cardinality {
                    constraint_id: constraint.constraint_id.clone(),
                    actual: identities.len(),
                });
            }
            let selection_digest = super::plan::canonical_digest(
                "resolved-constraint-target",
                &(
                    target,
                    Vec::<runmat_meshing_core::PersistentEntityId>::new(),
                    &identities,
                ),
            )?;
            ResolvedConstraintTarget {
                authored: target.clone(),
                matched_entities: Vec::new(),
                matched_node_identities: identities,
                selection_digest,
            }
        }
    };
    Ok((resolved, nodes))
}

fn available_dofs(
    topology: &CanonicalStructuralNodeSpace,
    node: usize,
) -> Vec<CanonicalStructuralDof> {
    let mut dofs = CanonicalStructuralDof::TRANSLATIONS.to_vec();
    if topology.nodes[node].owns_explicit_rotation {
        dofs.extend([
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Rx),
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Ry),
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Rz),
        ]);
    }
    dofs
}

fn visit_targets(
    kind: &StructuralConstraintKind,
    mut visit: impl FnMut(&runmat_analysis_core::RegionTarget),
) {
    match kind {
        StructuralConstraintKind::Fixed { target }
        | StructuralConstraintKind::NormalDisplacement { target, .. }
        | StructuralConstraintKind::Symmetry { target }
        | StructuralConstraintKind::Antisymmetry { target } => visit(target),
        StructuralConstraintKind::Prescribed(values) => visit(&values.target),
        StructuralConstraintKind::RigidCoupling(coupling)
        | StructuralConstraintKind::DistributingCoupling(coupling) => visit(&coupling.target),
        StructuralConstraintKind::Tie(pair) => {
            visit(&pair.primary);
            visit(&pair.secondary);
        }
        StructuralConstraintKind::Periodic(periodic) => {
            visit(&periodic.pair.primary);
            visit(&periodic.pair.secondary);
        }
        StructuralConstraintKind::Equation(equation) => {
            for term in &equation.terms {
                visit(&term.target);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AnalysisConstraint, CoordinateFrame, Displacement, StructuralConstraint,
        StructuralConstraintEquation, StructuralConstraintEquationTerm, StructuralConstraintKind,
        StructuralTranslationDof,
    };
    use runmat_meshing_core::{
        ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest,
    };

    use super::*;
    use crate::constraints::{CanonicalStructuralTopologyIdentity, ConstraintEnforcement};

    #[test]
    fn mixed_plan_round_trips_with_structural_rotation_rows() {
        let (model, mesh) = model_and_mesh();
        let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
        assert!(matches!(
            plan.topology,
            CanonicalStructuralTopologyIdentity::ConformalMixed { .. }
        ));
        assert!(plan.rows.iter().any(|row| {
            row.constraint_id == "beam_root_rotation"
                && row
                    .terms
                    .iter()
                    .any(|term| matches!(term.dof, CanonicalStructuralDof::Rotation(_)))
        }));
        let decoded = CanonicalStructuralConstraintPlan::from_canonical_bytes(
            &plan.canonical_bytes().unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, plan);
        decoded.validate_against(&model, &mesh).unwrap();
    }

    #[test]
    fn explicit_plan_round_trips_without_a_solver_mesh() {
        let model = crate::fixtures::fixture_model(
            crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
        );
        let plan = CanonicalStructuralConstraintPlan::compile_for_model(&model, None).unwrap();
        assert!(matches!(
            plan.topology,
            CanonicalStructuralTopologyIdentity::Explicit { .. }
        ));
        assert_eq!(plan.evidence.authored_constraint_count, 1);
        assert_eq!(plan.evidence.active_row_count, 6);
        plan.validate_against_explicit(&model).unwrap();
        let decoded = CanonicalStructuralConstraintPlan::from_canonical_bytes(
            &plan.canonical_bytes().unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, plan);
    }

    #[test]
    fn general_equation_spans_solid_and_explicit_nodes() {
        let (mut model, mesh) = model_and_mesh();
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: "solid_beam_ux_relation".into(),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Equation(StructuralConstraintEquation {
                    terms: vec![
                        StructuralConstraintEquationTerm {
                            target: runmat_analysis_core::RegionTarget::point("node:1"),
                            dof: StructuralTranslationDof::Ux,
                            coefficient: 1.0,
                        },
                        StructuralConstraintEquationTerm {
                            target: runmat_analysis_core::RegionTarget::point("node:102"),
                            dof: StructuralTranslationDof::Ux,
                            coefficient: -1.0,
                        },
                    ],
                    right_hand_side: Displacement::new(0.0),
                }),
            }));
        let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
        let row = plan
            .rows
            .iter()
            .find(|row| row.constraint_id == "solid_beam_ux_relation")
            .unwrap();
        assert_eq!(row.terms.len(), 2);
        assert_eq!(plan.enforcement, ConstraintEnforcement::AffineReduction);
        let result = crate::run_linear_static_with_options(
            &model,
            crate::ComputeBackend::Cpu,
            crate::LinearStaticSolveOptions {
                formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
                solver_mesh: Some(mesh),
                ..crate::LinearStaticSolveOptions::default()
            },
        )
        .unwrap();
        assert!(result
            .field(crate::FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE)
            .is_some());
    }

    fn model_and_mesh() -> (AnalysisModel, SolverMeshArtifact) {
        let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
        let attached_coordinates = mesh.topology.nodes[0].coordinates_m;
        mesh.topology.nodes[0].provenance.push(PersistentEntityId {
            kind: PersistentEntityKind::Vertex,
            source_topology_id: "beam_root".into(),
            assembly_path: vec!["root".into()],
        });
        mesh.topology.nodes[0].provenance.sort();
        for face in &mut mesh.topology.boundary_faces {
            face.provenance.push(PersistentEntityId {
                kind: PersistentEntityKind::Face,
                source_topology_id: "root".into(),
                assembly_path: vec!["root".into()],
            });
            face.provenance.sort();
        }
        mesh.topology.boundary_faces[1]
            .provenance
            .push(PersistentEntityId {
                kind: PersistentEntityKind::Face,
                source_topology_id: "solid_load".into(),
                assembly_path: vec!["root".into()],
            });
        mesh.topology.boundary_faces[1].provenance.sort();
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest().unwrap();
        (
            crate::fixtures::mixed_solid_beam_model(attached_coordinates),
            mesh,
        )
    }
}
