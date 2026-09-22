use runmat_analysis_core::{
    ExcitationDistribution, ExcitationKinematics, StructuralRotationDof, StructuralTranslationDof,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    assembly::{
        dofs::{StructuralDofKind, StructuralDofLayout},
        explicit_loads::apply_projected_explicit_structural_loads,
        solver_solid_boundary::apply_solver_mesh_structural_loads,
        MomentProjectionSummary,
    },
    constraints::{
        CanonicalStructuralConstraintPlan, CanonicalStructuralDof, CanonicalStructuralDofOwner,
    },
    excitation::{RealizedExcitation, RealizedExcitations, SpatialProjectionEvidence},
    physics::structural::topology::{
        node_space::CanonicalStructuralNodeSpace,
        target::{resolve_mixed_structural_target, MixedStructuralTargetOwnership},
    },
    properties::CompiledPropertyMap,
};

use super::{
    CanonicalStructuralLoadEntry, CanonicalStructuralLoadPlan, CanonicalStructuralLoadTerm,
    StructuralLoadPlanError,
};

const SAMPLE_PREFIX: &[u8] = b"runmat-fea-structural-sampled-value/v1\0";
const SAMPLE_DOMAIN: &str = "fea-structural-sampled-value";
const SAMPLE_LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 1024, 1024, 32);

pub(crate) struct StructuralLoadCompilation {
    pub plan: CanonicalStructuralLoadPlan,
    pub direct_rotational_moment_count: usize,
    pub moment_projection: Vec<MomentProjectionSummary>,
}

pub(crate) fn compile_solid(
    excitations: &RealizedExcitations,
    mesh: &SolverMeshArtifact,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    topology: &CanonicalStructuralNodeSpace,
    layout: &StructuralDofLayout,
) -> Result<StructuralLoadCompilation, StructuralLoadPlanError> {
    compile_entries(
        excitations,
        topology,
        &constraint_plan.reference_points,
        layout,
        |load, scratch| {
            validate_policy(load)?;
            let projection = apply_solver_mesh_structural_loads(
                &RealizedExcitations::new(vec![load.clone()]),
                mesh,
                constraint_plan,
                layout,
                scratch,
                true,
            )
            .map_err(|error| unresolved(load, error.to_string()))?;
            Ok(ProjectedLoad {
                evidence: require_evidence(load, projection.evidence)?,
                direct_rotational_moment_count: 0,
                moment_projection: projection.moment_projection,
            })
        },
    )
}

pub(crate) fn compile_explicit(
    structural: &runmat_analysis_core::StructuralModel,
    properties: &CompiledPropertyMap,
    excitations: &RealizedExcitations,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    topology: &CanonicalStructuralNodeSpace,
    layout: &StructuralDofLayout,
) -> Result<StructuralLoadCompilation, StructuralLoadPlanError> {
    compile_entries(
        excitations,
        topology,
        &constraint_plan.reference_points,
        layout,
        |load, scratch| {
            validate_policy(load)?;
            let projection = apply_projected_explicit_structural_loads(
                structural,
                properties,
                &RealizedExcitations::new(vec![load.clone()]),
                layout,
                scratch,
                topology,
            )
            .map_err(|error| unresolved(load, error.to_string()))?;
            Ok(ProjectedLoad {
                evidence: require_evidence(load, projection.evidence)?,
                direct_rotational_moment_count: projection.direct_rotational_moment_count,
                moment_projection: projection.moment_projection,
            })
        },
    )
}

pub(crate) fn compile_mixed(
    structural: &runmat_analysis_core::StructuralModel,
    properties: &CompiledPropertyMap,
    excitations: &RealizedExcitations,
    mesh: &SolverMeshArtifact,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    topology: &CanonicalStructuralNodeSpace,
    layout: &StructuralDofLayout,
) -> Result<StructuralLoadCompilation, StructuralLoadPlanError> {
    compile_entries(
        excitations,
        topology,
        &constraint_plan.reference_points,
        layout,
        |load, scratch| {
            validate_policy(load)?;
            if matches!(
                load.kind,
                crate::excitation::RealizedExcitationKind::BodyForce(_)
            ) && matches!(
                load.target,
                crate::excitation::RealizedExcitationTarget::WholeModel
            ) {
                let singleton = RealizedExcitations::new(vec![load.clone()]);
                let solid = apply_solver_mesh_structural_loads(
                    &singleton,
                    mesh,
                    constraint_plan,
                    layout,
                    scratch,
                    true,
                )
                .map_err(|error| unresolved(load, error.to_string()))?;
                let explicit = apply_projected_explicit_structural_loads(
                    structural, properties, &singleton, layout, scratch, topology,
                )
                .map_err(|error| unresolved(load, error.to_string()))?;
                let evidence = solid
                    .evidence
                    .into_iter()
                    .chain(explicit.evidence)
                    .collect();
                return Ok(ProjectedLoad {
                    evidence: require_evidence(load, evidence)?,
                    direct_rotational_moment_count: 0,
                    moment_projection: Vec::new(),
                });
            }
            let ownership = match &load.target {
                crate::excitation::RealizedExcitationTarget::Region(target) => {
                    resolve_mixed_structural_target(target, mesh, structural, topology)
                        .map_err(|error| unresolved(load, error.to_string()))?
                }
                crate::excitation::RealizedExcitationTarget::ReferencePoint { .. } => {
                    MixedStructuralTargetOwnership::Solid {
                        canonical_nodes: Vec::new(),
                    }
                }
                _ => {
                    return Err(unresolved(
                        load,
                        "mixed structural loads require a region or coupled reference point",
                    ));
                }
            };
            if matches!(
                ownership,
                MixedStructuralTargetOwnership::Explicit { .. }
                    | MixedStructuralTargetOwnership::Shared { .. }
            ) {
                let projection = apply_projected_explicit_structural_loads(
                    structural,
                    properties,
                    &RealizedExcitations::new(vec![load.clone()]),
                    layout,
                    scratch,
                    topology,
                )
                .map_err(|error| unresolved(load, error.to_string()))?;
                Ok(ProjectedLoad {
                    evidence: require_evidence(load, projection.evidence)?,
                    direct_rotational_moment_count: projection.direct_rotational_moment_count,
                    moment_projection: projection.moment_projection,
                })
            } else {
                let projection = apply_solver_mesh_structural_loads(
                    &RealizedExcitations::new(vec![load.clone()]),
                    mesh,
                    constraint_plan,
                    layout,
                    scratch,
                    true,
                )
                .map_err(|error| unresolved(load, error.to_string()))?;
                Ok(ProjectedLoad {
                    evidence: require_evidence(load, projection.evidence)?,
                    direct_rotational_moment_count: 0,
                    moment_projection: projection.moment_projection,
                })
            }
        },
    )
}

struct ProjectedLoad {
    evidence: Vec<SpatialProjectionEvidence>,
    direct_rotational_moment_count: usize,
    moment_projection: Vec<MomentProjectionSummary>,
}

fn compile_entries(
    excitations: &RealizedExcitations,
    topology: &CanonicalStructuralNodeSpace,
    reference_points: &[crate::constraints::CanonicalStructuralReferencePoint],
    layout: &StructuralDofLayout,
    mut project: impl FnMut(
        &RealizedExcitation,
        &mut [f64],
    ) -> Result<ProjectedLoad, StructuralLoadPlanError>,
) -> Result<StructuralLoadCompilation, StructuralLoadPlanError> {
    let mut entries = Vec::with_capacity(excitations.len());
    let mut direct_rotational_moment_count = 0;
    let mut moment_projection = Vec::new();
    for load in excitations.iter() {
        if !is_structural_load(&load.kind) {
            continue;
        }
        let mut scratch = vec![0.0; layout.total_dof_count()];
        let projected = project(load, &mut scratch)?;
        let terms = terms_from_rows(topology, reference_points, layout, &scratch, load)?;
        direct_rotational_moment_count += projected.direct_rotational_moment_count;
        moment_projection.extend(projected.moment_projection);
        entries.push(CanonicalStructuralLoadEntry {
            excitation_id: load.excitation_id.0.clone(),
            case_ids: load.case_ids.iter().map(|id| id.0.clone()).collect(),
            signal_id: load.signal_id.0.clone(),
            sampled_value_digest: sampled_value_digest(load)?,
            sample: load.sample.clone(),
            target: load.target.clone(),
            quantity: load.quantity,
            frame: load.policy.frame.clone(),
            distribution: load.policy.distribution,
            kinematics: load.policy.kinematics,
            terms,
            evidence: projected.evidence,
        });
    }
    Ok(StructuralLoadCompilation {
        plan: CanonicalStructuralLoadPlan::new(topology.identity, entries)?,
        direct_rotational_moment_count,
        moment_projection,
    })
}

fn is_structural_load(kind: &crate::excitation::RealizedExcitationKind) -> bool {
    matches!(
        kind,
        crate::excitation::RealizedExcitationKind::Force(_)
            | crate::excitation::RealizedExcitationKind::Moment(_)
            | crate::excitation::RealizedExcitationKind::Traction(_)
            | crate::excitation::RealizedExcitationKind::Pressure(_)
            | crate::excitation::RealizedExcitationKind::BodyForce(_)
    )
}

fn terms_from_rows(
    topology: &CanonicalStructuralNodeSpace,
    reference_points: &[crate::constraints::CanonicalStructuralReferencePoint],
    layout: &StructuralDofLayout,
    values: &[f64],
    load: &RealizedExcitation,
) -> Result<Vec<CanonicalStructuralLoadTerm>, StructuralLoadPlanError> {
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value != 0.0)
        .map(|(row, value)| {
            let (owner, kind) = if let Some(address) = layout.address(row) {
                (
                    CanonicalStructuralDofOwner::StructuralNode(
                        topology.nodes[address.node_index].identity,
                    ),
                    address.kind,
                )
            } else {
                let relative = row.checked_sub(layout.node_dof_count()).ok_or_else(|| {
                    unresolved(load, "load row is outside the canonical structural layout")
                })?;
                let point = relative / StructuralDofKind::ORDER.len();
                let kind = StructuralDofKind::ORDER[relative % StructuralDofKind::ORDER.len()];
                let identity = reference_points
                    .get(point)
                    .ok_or_else(|| unresolved(load, "reference-point row is out of bounds"))?
                    .identity;
                (CanonicalStructuralDofOwner::ReferencePoint(identity), kind)
            };
            Ok(CanonicalStructuralLoadTerm {
                owner,
                dof: canonical_dof(kind),
                value,
            })
        })
        .collect()
}

fn canonical_dof(kind: StructuralDofKind) -> CanonicalStructuralDof {
    match kind {
        StructuralDofKind::Ux => CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux),
        StructuralDofKind::Uy => CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy),
        StructuralDofKind::Uz => CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz),
        StructuralDofKind::Rx => CanonicalStructuralDof::Rotation(StructuralRotationDof::Rx),
        StructuralDofKind::Ry => CanonicalStructuralDof::Rotation(StructuralRotationDof::Ry),
        StructuralDofKind::Rz => CanonicalStructuralDof::Rotation(StructuralRotationDof::Rz),
    }
}

fn validate_policy(load: &RealizedExcitation) -> Result<(), StructuralLoadPlanError> {
    if load.policy.kinematics != ExcitationKinematics::FixedFrame {
        return Err(StructuralLoadPlanError::UnsupportedPolicy {
            excitation_id: load.excitation_id.0.clone(),
            policy: "kinematics",
            value: "follower".into(),
        });
    }
    if load.policy.distribution == ExcitationDistribution::TopologyField {
        return Err(StructuralLoadPlanError::UnsupportedPolicy {
            excitation_id: load.excitation_id.0.clone(),
            policy: "distribution",
            value: "topology_field".into(),
        });
    }
    if matches!(
        load.policy.frame,
        runmat_analysis_core::CoordinateFrame::Material
            | runmat_analysis_core::CoordinateFrame::Element
    ) {
        return Err(StructuralLoadPlanError::UnsupportedPolicy {
            excitation_id: load.excitation_id.0.clone(),
            policy: "frame",
            value: load.policy.frame.context_label(),
        });
    }
    if matches!(
        load.kind,
        crate::excitation::RealizedExcitationKind::Pressure(_)
    ) && !matches!(
        load.policy.frame,
        runmat_analysis_core::CoordinateFrame::Global
            | runmat_analysis_core::CoordinateFrame::BoundaryNormalTangent
    ) {
        return Err(StructuralLoadPlanError::UnsupportedPolicy {
            excitation_id: load.excitation_id.0.clone(),
            policy: "pressure frame",
            value: load.policy.frame.context_label(),
        });
    }
    if matches!(
        load.kind,
        crate::excitation::RealizedExcitationKind::BodyForce(_)
    ) && load.policy.frame == runmat_analysis_core::CoordinateFrame::BoundaryNormalTangent
    {
        return Err(StructuralLoadPlanError::UnsupportedPolicy {
            excitation_id: load.excitation_id.0.clone(),
            policy: "body-force frame",
            value: load.policy.frame.context_label(),
        });
    }
    Ok(())
}

fn sampled_value_digest(load: &RealizedExcitation) -> Result<Digest, StructuralLoadPlanError> {
    encode_contract(
        SAMPLE_PREFIX,
        SAMPLE_DOMAIN,
        &(load.quantity, &load.sample, &load.kind),
        SAMPLE_LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|error| StructuralLoadPlanError::Encoding(error.to_string()))
}

fn require_evidence(
    load: &RealizedExcitation,
    evidence: Vec<SpatialProjectionEvidence>,
) -> Result<Vec<SpatialProjectionEvidence>, StructuralLoadPlanError> {
    if evidence.is_empty() {
        return Err(StructuralLoadPlanError::InvalidProjection {
            excitation_id: load.excitation_id.0.clone(),
            reason: "projection did not produce evidence".into(),
        });
    }
    Ok(evidence)
}

fn unresolved(load: &RealizedExcitation, reason: impl Into<String>) -> StructuralLoadPlanError {
    StructuralLoadPlanError::UnresolvedTarget {
        excitation_id: load.excitation_id.0.clone(),
        reason: reason.into(),
    }
}
