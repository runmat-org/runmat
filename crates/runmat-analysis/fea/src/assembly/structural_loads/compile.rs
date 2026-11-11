use futures::FutureExt;
use runmat_analysis_core::{
    ExcitationDistribution, ExcitationKinematics, StructuralRotationDof, StructuralTranslationDof,
};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    assembly::{
        dofs::{StructuralDofKind, StructuralDofLayout},
        explicit_loads::apply_projected_explicit_structural_loads,
        solver_solid_boundary::apply_solver_mesh_structural_loads_on_projection,
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
    capacity::{admit_body_quadrature, derive_solid_projection, solid_load_error},
    CanonicalStructuralLoadEntry, CanonicalStructuralLoadPlan, CanonicalStructuralLoadTerm,
    MixedLoadCapacityError, MixedLoadPhase, StructuralLoadPlanError,
};

pub(crate) struct StructuralLoadCompilation {
    pub plan: CanonicalStructuralLoadPlan,
    pub direct_rotational_moment_count: usize,
    pub moment_projection: Vec<MomentProjectionSummary>,
}

pub(crate) struct MixedLoadContext<'a> {
    pub structural: &'a runmat_analysis_core::StructuralModel,
    pub properties: &'a CompiledPropertyMap,
    pub excitations: &'a RealizedExcitations,
    pub mesh: &'a SolverMeshArtifact,
    pub constraint_plan: &'a CanonicalStructuralConstraintPlan,
    pub topology: &'a CanonicalStructuralNodeSpace,
    pub layout: &'a StructuralDofLayout,
}

pub(crate) fn compile_solid(
    excitations: &RealizedExcitations,
    mesh: &SolverMeshArtifact,
    projection: &Result<crate::topology::AnalysisProjection, String>,
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
            let analysis_projection = projection.as_ref().map_err(|error| {
                unresolved(
                    load,
                    crate::assembly::SolverMeshRegionMappingError::InvalidAnalysisProjection(
                        error.clone(),
                    )
                    .to_string(),
                )
            })?;
            let projection = apply_solver_mesh_structural_loads_on_projection(
                &RealizedExcitations::new(vec![load.clone()]),
                mesh,
                analysis_projection,
                constraint_plan,
                layout,
                scratch,
                true,
            )
            .map_err(|error| solid_load_error(load, error))?;
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
) -> Result<StructuralLoadCompilation, MixedLoadCapacityError<std::convert::Infallible>> {
    compile_mixed_with_owner(
        MixedLoadContext {
            structural,
            properties,
            excitations,
            mesh,
            constraint_plan,
            topology,
            layout,
        },
        false,
        |_, _| std::future::ready(Ok::<(), std::convert::Infallible>(())),
    )
    .now_or_never()
    .ok_or_else(|| {
        MixedLoadCapacityError::Plan(StructuralLoadPlanError::Invalid(
            "synchronous mixed-load projection yielded without an admission owner".into(),
        ))
    })?
}

pub(crate) async fn compile_mixed_with_owner<O, F, A>(
    context: MixedLoadContext<'_>,
    awaited_elements: bool,
    mut before: O,
) -> Result<StructuralLoadCompilation, MixedLoadCapacityError<A>>
where
    O: FnMut(MixedLoadPhase, runmat_simulation_usage::OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    let MixedLoadContext {
        structural,
        properties,
        excitations,
        mesh,
        constraint_plan,
        topology,
        layout,
    } = context;
    let mut entries = EntryAccumulator::new(
        topology,
        &constraint_plan.reference_points,
        layout,
        excitations.len(),
    );
    for load in excitations.iter() {
        if !load.kind.is_structural_load() {
            continue;
        }
        let mut scratch = vec![0.0; layout.total_dof_count()];
        let projected = async {
            validate_policy(load).map_err(MixedLoadCapacityError::Plan)?;
            if matches!(
                load.kind,
                crate::excitation::RealizedExcitationKind::BodyForce(_)
            ) && matches!(
                load.target,
                crate::excitation::RealizedExcitationTarget::WholeModel
            ) {
                let singleton = RealizedExcitations::new(vec![load.clone()]);
                let projection =
                    derive_solid_projection(mesh, awaited_elements, &mut before).await?;
                admit_body_quadrature(&projection, load, awaited_elements, &mut before).await?;
                let solid = apply_solver_mesh_structural_loads_on_projection(
                    &singleton,
                    mesh,
                    &projection,
                    constraint_plan,
                    layout,
                    &mut scratch,
                    true,
                )
                .map_err(|error| MixedLoadCapacityError::Plan(solid_load_error(load, error)))?;
                let explicit = apply_projected_explicit_structural_loads(
                    structural,
                    properties,
                    &singleton,
                    layout,
                    &mut scratch,
                    topology,
                )
                .map_err(|error| {
                    MixedLoadCapacityError::Plan(unresolved(load, error.to_string()))
                })?;
                let evidence = solid
                    .evidence
                    .into_iter()
                    .chain(explicit.evidence)
                    .collect();
                return Ok::<_, MixedLoadCapacityError<A>>(ProjectedLoad {
                    evidence: require_evidence(load, evidence)
                        .map_err(MixedLoadCapacityError::Plan)?,
                    direct_rotational_moment_count: 0,
                    moment_projection: Vec::new(),
                });
            }
            let ownership = match &load.target {
                crate::excitation::RealizedExcitationTarget::Region(target) => {
                    resolve_mixed_structural_target(target, mesh, structural, topology).map_err(
                        |error| MixedLoadCapacityError::Plan(unresolved(load, error.to_string())),
                    )?
                }
                crate::excitation::RealizedExcitationTarget::ReferencePoint { .. } => {
                    MixedStructuralTargetOwnership::Solid {
                        canonical_nodes: Vec::new(),
                    }
                }
                _ => {
                    return Err(MixedLoadCapacityError::Plan(unresolved(
                        load,
                        "mixed structural loads require a region or coupled reference point",
                    )));
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
                    &mut scratch,
                    topology,
                )
                .map_err(|error| {
                    MixedLoadCapacityError::Plan(unresolved(load, error.to_string()))
                })?;
                Ok(ProjectedLoad {
                    evidence: require_evidence(load, projection.evidence)
                        .map_err(MixedLoadCapacityError::Plan)?,
                    direct_rotational_moment_count: projection.direct_rotational_moment_count,
                    moment_projection: projection.moment_projection,
                })
            } else {
                let solid_projection =
                    derive_solid_projection(mesh, awaited_elements, &mut before).await?;
                admit_body_quadrature(&solid_projection, load, awaited_elements, &mut before)
                    .await?;
                let projection = apply_solver_mesh_structural_loads_on_projection(
                    &RealizedExcitations::new(vec![load.clone()]),
                    mesh,
                    &solid_projection,
                    constraint_plan,
                    layout,
                    &mut scratch,
                    true,
                )
                .map_err(|error| MixedLoadCapacityError::Plan(solid_load_error(load, error)))?;
                Ok(ProjectedLoad {
                    evidence: require_evidence(load, projection.evidence)
                        .map_err(MixedLoadCapacityError::Plan)?,
                    direct_rotational_moment_count: 0,
                    moment_projection: projection.moment_projection,
                })
            }
        }
        .await?;
        entries
            .append(load, &scratch, projected)
            .map_err(MixedLoadCapacityError::Plan)?;
    }
    entries.finish().map_err(MixedLoadCapacityError::Plan)
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
    let mut entries = EntryAccumulator::new(topology, reference_points, layout, excitations.len());
    for load in excitations.iter() {
        if !load.kind.is_structural_load() {
            continue;
        }
        let mut scratch = vec![0.0; layout.total_dof_count()];
        let projected = project(load, &mut scratch)?;
        entries.append(load, &scratch, projected)?;
    }
    entries.finish()
}

struct EntryAccumulator<'a> {
    topology: &'a CanonicalStructuralNodeSpace,
    reference_points: &'a [crate::constraints::CanonicalStructuralReferencePoint],
    layout: &'a StructuralDofLayout,
    entries: Vec<CanonicalStructuralLoadEntry>,
    direct_rotational_moment_count: usize,
    moment_projection: Vec<MomentProjectionSummary>,
}

impl<'a> EntryAccumulator<'a> {
    fn new(
        topology: &'a CanonicalStructuralNodeSpace,
        reference_points: &'a [crate::constraints::CanonicalStructuralReferencePoint],
        layout: &'a StructuralDofLayout,
        capacity: usize,
    ) -> Self {
        Self {
            topology,
            reference_points,
            layout,
            entries: Vec::with_capacity(capacity),
            direct_rotational_moment_count: 0,
            moment_projection: Vec::new(),
        }
    }

    fn append(
        &mut self,
        load: &RealizedExcitation,
        scratch: &[f64],
        projected: ProjectedLoad,
    ) -> Result<(), StructuralLoadPlanError> {
        let terms = terms_from_rows(
            self.topology,
            self.reference_points,
            self.layout,
            scratch,
            load,
        )?;
        self.direct_rotational_moment_count += projected.direct_rotational_moment_count;
        self.moment_projection.extend(projected.moment_projection);
        self.entries.push(CanonicalStructuralLoadEntry {
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
        Ok(())
    }

    fn finish(self) -> Result<StructuralLoadCompilation, StructuralLoadPlanError> {
        Ok(StructuralLoadCompilation {
            plan: CanonicalStructuralLoadPlan::new(self.topology.identity, self.entries)?,
            direct_rotational_moment_count: self.direct_rotational_moment_count,
            moment_projection: self.moment_projection,
        })
    }
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
    let nodal_body_field = matches!(
        load.kind,
        crate::excitation::RealizedExcitationKind::BodyForceField(_)
    );
    if (load.policy.distribution == ExcitationDistribution::TopologyField) != nodal_body_field {
        return Err(StructuralLoadPlanError::UnsupportedPolicy {
            excitation_id: load.excitation_id.0.clone(),
            policy: "distribution",
            value: "topology_field is required exclusively for a spatial body-density field".into(),
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
            | crate::excitation::RealizedExcitationKind::BodyForceField(_)
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
    crate::excitation::sample_identity::sampled_value_digest(load)
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
