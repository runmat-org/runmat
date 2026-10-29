//! Authored contact interfaces realized as the penalty/Coulomb contact operator of a nonlinear
//! structural step. The interfaces, surfaces and friction law come from the model; the
//! enforcement controls scale the penalty and bound the admitted initial penetration.
use super::*;
use crate::{
    assembly::{dofs::StructuralDofKind, AssemblySummary},
    physics::structural::{
        contact::{
            CanonicalContactOperator, ContactInterfacePolicy, ContactOperatorPolicy,
            ContactPenaltyPolicy,
        },
        topology::node_space::CanonicalStructuralNodeSpace,
    },
    solve::{
        general::{GeneralSolveEnvelope, GeneralSolvePolicy},
        nonlinear::{solid::SolidContactCoupling, ContactBlockExecution},
    },
    topology::{
        AnalysisProjection, BoundaryPairingPolicy, BoundaryProjectionDifferentialPolicy,
        BoundarySearchPolicy, BoundaryStationaryPolicy,
    },
    FeaContactEnforcement,
};
use runmat_analysis_core::{AnalysisInterfaceKind, ContactFriction};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeMap;

/// Penalty normal stiffness per unit area at scale one: `10 E_min / h`.
const NORMAL_PENALTY_FACTOR: f64 = 10.0;
/// Pairing search reaches this many face sizes beyond the primary surface.
const SEARCH_FACE_SIZES: f64 = 4.0;

/// Owned inputs that the contact operator borrows for the whole step.
pub(super) struct ContactInputs {
    pub projection: AnalysisProjection,
    pub node_rows: Vec<[usize; 3]>,
    pub controls: Vec<ContactInterfacePolicy>,
    pub budget: ContactOperatorPolicy,
}

pub(super) fn prepare_inputs(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    summary: &AssemblySummary,
    enforcement: FeaContactEnforcement,
    maximum_resident_bytes: u64,
) -> Result<Option<ContactInputs>, FeaRunError> {
    if model.interfaces.is_empty() {
        return Ok(None);
    }
    if model
        .interfaces
        .iter()
        .any(|interface| !matches!(interface.kind, AnalysisInterfaceKind::Contact(_)))
    {
        return Err(FeaRunError::InvalidModel(
            "structural nonlinear realizes contact interfaces only".into(),
        ));
    }
    if !enforcement.penalty_stiffness_scale.is_finite()
        || enforcement.penalty_stiffness_scale <= 0.0
        || !enforcement.max_penetration_ratio.is_finite()
        || enforcement.max_penetration_ratio < 0.0
    {
        return Err(FeaRunError::InvalidModel(
            "contact enforcement needs a positive penalty scale and a nonnegative penetration ratio"
                .into(),
        ));
    }
    let projection =
        AnalysisProjection::derive(mesh).map_err(|e| FeaRunError::Assembly(e.to_string()))?;
    let nodes = CanonicalStructuralNodeSpace::derive_solid(mesh)
        .map_err(|e| FeaRunError::Assembly(e.to_string()))?;
    let canonical = nodes
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index))
        .collect::<BTreeMap<_, _>>();
    let mut rows_by_node = vec![[usize::MAX; 3]; nodes.nodes.len()];
    let layout = &summary.structural_dof_layout;
    for (row, address) in (0..layout.node_dof_count())
        .filter_map(|row| layout.address(row).map(|address| (row, address)))
    {
        let axis = match address.kind {
            StructuralDofKind::Ux => 0,
            StructuralDofKind::Uy => 1,
            StructuralDofKind::Uz => 2,
            _ => continue,
        };
        if let Some(slot) = rows_by_node.get_mut(address.node_index) {
            slot[axis] = row;
        }
    }
    let node_rows = projection
        .nodes
        .iter()
        .map(|node| {
            canonical
                .get(&node.stable_identity)
                .map(|&index| rows_by_node[index])
                .filter(|rows| rows.iter().all(|row| *row != usize::MAX))
                .ok_or_else(|| FeaRunError::Assembly("contact node has no translation rows".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let youngs = model
        .materials
        .iter()
        .filter_map(|material| material.require_mechanical().ok())
        .map(|mechanical| mechanical.youngs_modulus_pa.canonical_value())
        .fold(f64::INFINITY, f64::min);
    if !youngs.is_finite() || youngs <= 0.0 {
        return Err(FeaRunError::InvalidModel(
            "contact penalty needs an elastic material".into(),
        ));
    }
    let mut controls = Vec::with_capacity(model.interfaces.len());
    for interface in &model.interfaces {
        let AnalysisInterfaceKind::Contact(contact) = &interface.kind else {
            unreachable!("checked above");
        };
        let (area, count) = projection
            .boundary_faces
            .iter()
            .filter(|face| {
                face.provenance.iter().any(|entity| {
                    interface.secondary.matches_persistent_entity(entity)
                        || interface.primary.matches_persistent_entity(entity)
                })
            })
            .fold((0.0, 0usize), |(area, count), face| {
                (area + face.area_m2.canonical_value(), count + 1)
            });
        if count == 0 || !(area > 0.0) {
            return Err(FeaRunError::InvalidModel(format!(
                "contact interface `{}` selects no boundary faces",
                interface.interface_id
            )));
        }
        // Mean triangle edge length of the paired surfaces.
        let size = (4.0 * area / (count as f64 * 3.0_f64.sqrt())).sqrt();
        let normal = enforcement.penalty_stiffness_scale * NORMAL_PENALTY_FACTOR * youngs / size;
        controls.push(ContactInterfacePolicy {
            penalty: ContactPenaltyPolicy {
                normal_stiffness_pa_per_m:
                    runmat_types::physical::TractionDisplacementStiffness::new(normal),
                tangential_stiffness_pa_per_m: match contact.friction {
                    ContactFriction::Frictionless => None,
                    ContactFriction::Coulomb { .. } => Some(
                        runmat_types::physical::TractionDisplacementStiffness::new(normal),
                    ),
                },
            },
            pairing: BoundaryPairingPolicy {
                narrow: BoundarySearchPolicy {
                    distance_tolerance_m: 1e-9 * size,
                    maximum_patches: 1 << 20,
                    maximum_depth: 24,
                },
                maximum_distance_m: SEARCH_FACE_SIZES * size,
                distance_separation_m: 1e-6 * size,
                maximum_faces: 16,
                maximum_total_patches: 1 << 24,
            },
            stationary: BoundaryStationaryPolicy {
                differential: BoundaryProjectionDifferentialPolicy {
                    stationarity_tolerance_m: 1e-10 * size,
                    minimum_barycentric: 1e-7,
                    minimum_metric_ratio: 1e-10,
                },
                maximum_iterations: 24,
                maximum_backtracks: 24,
            },
            maximum_initial_penetration_m: enforcement.max_penetration_ratio * size,
        });
    }
    let nodes_total = projection.nodes.len();
    let faces_total = projection.boundary_faces.len();
    let budget = ContactOperatorPolicy {
        maximum_dofs: summary.dof_count,
        maximum_reference_points: 64 * faces_total.max(1),
        maximum_face_pairs: 64 * faces_total.max(1),
        maximum_coordinates: 3 * nodes_total.max(1) * 4,
        maximum_patches_per_trial: 1 << 32,
        maximum_refinement_evaluations_per_trial: 1 << 32,
        maximum_preparation_resident_bytes: maximum_resident_bytes,
        maximum_preparation_scalar_operations: 1 << 40,
    };
    Ok(Some(ContactInputs {
        projection,
        node_rows,
        controls,
        budget,
    }))
}

pub(super) fn prepare_operator<'a>(
    model: &'a AnalysisModel,
    inputs: &'a ContactInputs,
    dof_count: usize,
) -> Result<CanonicalContactOperator<'a>, FeaRunError> {
    CanonicalContactOperator::prepare(
        &inputs.projection,
        &model.interfaces,
        &inputs.node_rows,
        dof_count,
        &inputs.controls,
        inputs.budget,
    )
    .map_err(|error| FeaRunError::InvalidModel(format!("contact preparation failed: {error}")))
}

pub(super) fn execution<'a>(
    operator: &'a CanonicalContactOperator<'a>,
    options: &NonlinearSolveOptions,
    maximum_resident_bytes: u64,
) -> ContactBlockExecution<'a> {
    let solve_policy = GeneralSolvePolicy::bounded(options.linear_tolerance);
    ContactBlockExecution {
        coupling: SolidContactCoupling {
            operator,
            solve_policy,
            solve_admission: GeneralSolveEnvelope {
                resident_bytes: maximum_resident_bytes,
                scalar_operations: u64::MAX,
            },
            maximum_additional_resident_bytes: maximum_resident_bytes,
            maximum_geometry_trials_per_advance: u64::MAX,
            maximum_symbolic_scalar_operations: u64::MAX,
        },
        maximum_retained_output_bytes: maximum_resident_bytes,
    }
}
