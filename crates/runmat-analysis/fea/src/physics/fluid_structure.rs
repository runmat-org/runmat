//! Two-way fluid–structure interaction: the canonical finite-volume flow on the deformed fluid
//! domain loads a small-strain elastic structure, and the structure's interface displacement
//! moves the fluid wall.
//!
//! Each coupling iteration solves the flow on the current deformed configuration, integrates
//! the pressure and viscous traction of every interface face onto the structure's interface
//! nodes (the exact negative of the force the wall exerts on the fluid), solves the structure,
//! and relaxes the interface displacement with Aitken's dynamic factor. The fluid interior
//! follows the wall through a volume-weighted Laplace extension. Iteration stops when the
//! structure's response to the retained flow reproduces the displacement that shaped it.
//!
//! Retained evidence carries every iteration's interface residual, the converged traction
//! resultant, the work of that traction on the converged displacement and the structural
//! strain energy (a linear structure loaded only at the interface stores half of that work),
//! and the equilibrium residuals of the structure and the mesh motion.

mod mesh_motion;
mod structure;
mod transient;
pub use structure::NewmarkState;
pub use transient::{
    FluidStructureFlowState, FluidStructureStepEvidence, FluidStructureTransientResult,
};
#[cfg(test)]
mod tests;

use crate::{
    algebra::RealSparseMatrix,
    physics::cfd::canonical::{
        CanonicalFluidError, CanonicalFluidInput, CanonicalFluidPolicy, CanonicalFluidProblem,
        CanonicalFluidResult,
    },
    properties::CompiledPropertyMap,
    solve::shared_sparse::{ambient_plan, control, with_plan, CsrArrays},
    topology::{
        AnalysisProjection, FiniteVolumeCellSelection, FiniteVolumeProjectionPolicy,
        FiniteVolumeTopology,
    },
};
use runmat_analysis_core::{
    AnalysisBoundary, AnalysisInterfaceKind, AnalysisModel, CfdDomain, ExcitationAction,
    FluidConstraint, FluidConstraintKind, PhysicalSignal,
};
use runmat_meshing_core::{BoundaryFaceRole, SolverMeshArtifact};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FluidStructureError {
    #[error("invalid fluid–structure input: {0}")]
    Invalid(&'static str),
    #[error("invalid fluid–structure input: {0}")]
    InvalidDetail(String),
    #[error("unsupported fluid–structure envelope: {0}")]
    Unsupported(&'static str),
    #[error("fluid–structure sparse operator failed: {0}")]
    Operator(String),
    #[error("fluid–structure execution cancelled")]
    Cancelled,
    #[error(transparent)]
    Flow(#[from] CanonicalFluidError),
    #[error(transparent)]
    Admission(#[from] crate::admission::AdmissionIssue),
    #[error("fluid–structure linear solve did not converge: iterations={iterations}, relative_residual={relative_residual:?}")]
    LinearNotConverged {
        iterations: u32,
        relative_residual: Option<f64>,
    },
    #[error("fluid–structure coupling did not converge in {iterations} iterations; interface residuals {residuals:?}")]
    CouplingNotConverged {
        iterations: usize,
        residuals: Vec<f64>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidStructurePolicy {
    /// Interface displacement residual relative to the interface displacement.
    pub coupling_relative_tolerance: f64,
    pub maximum_coupling_iterations: usize,
    /// First relaxation factor before Aitken's update takes over.
    pub initial_relaxation: f64,
}

impl Default for FluidStructurePolicy {
    fn default() -> Self {
        Self {
            coupling_relative_tolerance: 1e-6,
            maximum_coupling_iterations: 60,
            initial_relaxation: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LinearPolicy {
    /// Relative residual the refined direct solve must reach on the original operator.
    pub relative_tolerance: f64,
}

/// Borrowed authored flow authority, as compiled by the fluid source owner.
pub struct FluidAuthority<'a> {
    pub material: &'a CfdDomain,
    pub constraints: &'a [FluidConstraint],
    pub boundary_operators: &'a [AnalysisBoundary],
    pub signals: &'a [PhysicalSignal],
    pub excitations: &'a [ExcitationAction],
    pub policy: CanonicalFluidPolicy,
}

pub struct FluidStructureInput<'a> {
    pub model: &'a AnalysisModel,
    /// Reference (undeformed) configuration.
    pub mesh: &'a SolverMeshArtifact,
    pub projection: &'a AnalysisProjection,
    pub properties: &'a CompiledPropertyMap,
    pub fluid_selection: &'a FiniteVolumeCellSelection,
    pub geometry_policy: FiniteVolumeProjectionPolicy,
    pub fluid: FluidAuthority<'a>,
    pub policy: FluidStructurePolicy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidStructureEvidence {
    /// Relative interface displacement change at every coupling iteration.
    pub iteration_residuals: Vec<f64>,
    /// |response − retained| / |response| on the interface for the retained flow.
    pub interface_displacement_relative_residual: f64,
    /// Resultant force of the fluid on the structure, N.
    pub interface_force_n: [f64; 3],
    /// Work of the converged interface traction on the structural response, J.
    pub interface_work_j: f64,
    /// ½ uᵀ K u of the structural response, J.
    pub structural_strain_energy_j: f64,
    /// |W − 2U| / W: Clapeyron's identity for a linear structure loaded at the interface.
    pub work_energy_relative_residual: f64,
    pub structural_equilibrium_relative_residual: f64,
    pub mesh_motion_relative_residual: f64,
    pub largest_interface_displacement_m: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidStructureResult {
    /// Displacement of every reference node: structural nodes carry the retained interface
    /// state, fluid-only nodes its mesh-motion extension. This shapes the retained flow.
    pub displacement_m: Vec<[f64; 3]>,
    /// The structure's response to the retained flow, one entry per structural node.
    pub structural_response_m: Vec<[f64; 3]>,
    /// Flow on the deformed configuration, in the reference cell and face order.
    pub flow: CanonicalFluidResult,
    pub coupling_iterations: usize,
    pub evidence: FluidStructureEvidence,
}

/// Prepared reference operators: the structure and the mesh motion do not change with the
/// small-strain deflection; only the fluid domain is re-derived on each configuration.
pub struct FluidStructureProblem<'a> {
    input: FluidStructureInput<'a>,
    structure: structure::ElasticStructure,
    motion: mesh_motion::MeshMotion,
    reference_fluid: FiniteVolumeTopology,
    /// (reference projection boundary face) of every fluid–structure interface face.
    interface_faces: Vec<usize>,
    interface_nodes: Vec<usize>,
    /// Nodes of rigid fluid walls that no coupled face shares: held in place for the fluid.
    rigid_wall_nodes: Vec<usize>,
    linear: LinearPolicy,
}

impl<'a> FluidStructureProblem<'a> {
    pub fn prepare(input: FluidStructureInput<'a>) -> Result<Self, FluidStructureError> {
        if !input.policy.coupling_relative_tolerance.is_finite()
            || input.policy.coupling_relative_tolerance <= 0.0
            || input.policy.maximum_coupling_iterations == 0
            || !(0.0..=1.0).contains(&input.policy.initial_relaxation)
            || input.policy.initial_relaxation == 0.0
        {
            return Err(FluidStructureError::Invalid("coupling policy"));
        }
        let projection = input.projection;
        let reference_fluid = FiniteVolumeTopology::derive_selected(
            projection,
            input.geometry_policy,
            input.fluid_selection.clone(),
        )
        .map_err(|e| FluidStructureError::InvalidDetail(e.to_string()))?;
        let structure =
            structure::ElasticStructure::assemble(input.model, projection, input.properties)?;
        let FluidSolidFaces {
            coupled: interface_faces,
            rigid,
        } = fluid_solid_faces(input.model, projection, &reference_fluid, &structure)?;
        let nodes_of = |faces: &[usize]| {
            let mut nodes = faces
                .iter()
                .flat_map(|&f| projection.boundary_faces[f].node_indices.iter().copied())
                .collect::<Vec<_>>();
            nodes.sort_unstable();
            nodes.dedup();
            nodes
        };
        let interface_nodes = nodes_of(&interface_faces);
        // A rigid wall stays in place for the fluid. A node it shares with a coupled face
        // follows the structure.
        let rigid_wall_nodes = nodes_of(&rigid)
            .into_iter()
            .filter(|n| interface_nodes.binary_search(n).is_err())
            .collect::<Vec<_>>();
        // Every node on the fluid's boundary has a prescribed motion: the interface nodes
        // move with the structure and all other fluid boundary nodes stay in place.
        let mut prescribed = vec![false; projection.nodes.len()];
        for face in &reference_fluid.faces {
            if face.neighbor.is_some() {
                continue;
            }
            let source = face.boundary_source.ok_or(FluidStructureError::Invalid(
                "fluid face without boundary source",
            ))?;
            for &node in &projection.boundary_faces[source.projection_boundary_index].node_indices {
                prescribed[node] = true;
            }
        }
        let motion = mesh_motion::MeshMotion::assemble(projection, &reference_fluid, &prescribed)?;
        let linear = LinearPolicy {
            relative_tolerance: input.fluid.policy.linear_relative_tolerance,
        };
        Ok(Self {
            input,
            structure,
            motion,
            reference_fluid,
            interface_faces,
            interface_nodes,
            rigid_wall_nodes,
            linear,
        })
    }

    pub fn reference_fluid(&self) -> &FiniteVolumeTopology {
        &self.reference_fluid
    }

    pub fn structural_node_count(&self) -> usize {
        self.structure.nodes.len()
    }

    pub fn maximum_coupling_iterations(&self) -> usize {
        self.input.policy.maximum_coupling_iterations
    }

    pub fn solve(&self) -> Result<FluidStructureResult, FluidStructureError> {
        let n_struct = self.structure.dofs();
        let mut interface_state = vec![0.0; n_struct];
        let mut previous_residual: Option<Vec<f64>> = None;
        let mut relaxation = self.input.policy.initial_relaxation;
        let mut residuals = Vec::new();
        let mut warm: Option<CanonicalFluidResult> = None;
        for iteration in 1..=self.input.policy.maximum_coupling_iterations {
            if crate::is_fea_cancelled() {
                return Err(FluidStructureError::Cancelled);
            }
            let displacement = self.configuration(&interface_state)?;
            let deformed = self.deformed_mesh(&displacement)?;
            let projection = AnalysisProjection::derive(&deformed)
                .map_err(|e| FluidStructureError::InvalidDetail(e.to_string()))?;
            let topology = self.fluid_topology(&projection)?;
            let problem = self.flow_problem(&projection, &topology)?;
            let flow = match &warm {
                Some(previous) => problem.solve_from(previous.original_fields())?,
                None => problem.solve()?,
            };
            let load = self.interface_load(&topology, &flow)?;
            let (response, _) = self.structure.solve(&load, self.linear)?;
            let residual = response
                .iter()
                .zip(&interface_state)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            let scale = self.interface_norm(&response).max(f64::MIN_POSITIVE);
            let relative = self.interface_norm(&residual) / scale;
            residuals.push(
                if self.interface_norm(&response) == 0.0 && self.interface_norm(&residual) == 0.0 {
                    0.0
                } else {
                    relative
                },
            );
            if *residuals.last().unwrap() <= self.input.policy.coupling_relative_tolerance {
                let evidence = self.evidence(&displacement, &response, &load, &flow, residuals)?;
                return Ok(FluidStructureResult {
                    displacement_m: displacement,
                    structural_response_m: to_vectors(&response),
                    flow,
                    coupling_iterations: iteration,
                    evidence,
                });
            }
            // Aitken's dynamic relaxation on the interface displacement residual.
            if let Some(previous) = &previous_residual {
                let delta = residual
                    .iter()
                    .zip(previous)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>();
                let denominator = dot(&delta, &delta);
                if denominator > 0.0 {
                    relaxation =
                        (-relaxation * dot(previous, &delta) / denominator).clamp(-1.0, 1.0);
                    if relaxation.abs() < 1e-3 {
                        relaxation = 1e-3f64.copysign(relaxation);
                    }
                }
            }
            for (state, r) in interface_state.iter_mut().zip(&residual) {
                *state += relaxation * r;
            }
            previous_residual = Some(residual);
            warm = Some(flow);
        }
        Err(FluidStructureError::CouplingNotConverged {
            iterations: self.input.policy.maximum_coupling_iterations,
            residuals,
        })
    }

    /// Re-evaluate every retained equation: the flow on the retained configuration, the mesh
    /// motion, structural equilibrium under the retained flow and the coupling residual.
    pub fn recover(&self, original: &FluidStructureResult) -> Result<(), FluidStructureError> {
        let p = self.input.projection;
        if original.displacement_m.len() != p.nodes.len()
            || original.structural_response_m.len() != self.structure.nodes.len()
        {
            return Err(FluidStructureError::Invalid(
                "retained displacement inventory",
            ));
        }
        let mut state = vec![0.0; self.structure.dofs()];
        for (ordinal, &node) in self.structure.nodes.iter().enumerate() {
            state[3 * ordinal..3 * ordinal + 3].copy_from_slice(&original.displacement_m[node]);
        }
        let configuration = self.configuration(&state)?;
        if configuration
            .iter()
            .zip(&original.displacement_m)
            .any(|(a, b)| {
                (0..3).any(|k| {
                    (a[k] - b[k]).abs()
                        > 1e-9
                            * (b[k].abs()
                                + self.characteristic_displacement(&original.displacement_m))
                })
            })
        {
            return Err(FluidStructureError::Invalid(
                "retained fluid mesh motion differs from its interface displacement",
            ));
        }
        let deformed = self.deformed_mesh(&original.displacement_m)?;
        let projection = AnalysisProjection::derive(&deformed)
            .map_err(|e| FluidStructureError::InvalidDetail(e.to_string()))?;
        let topology = self.fluid_topology(&projection)?;
        let problem = self.flow_problem(&projection, &topology)?;
        problem.recover(&original.flow)?;
        let load = self.interface_load(&topology, &original.flow)?;
        let response = original
            .structural_response_m
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect::<Vec<_>>();
        let evidence = self.evidence(
            &original.displacement_m,
            &response,
            &load,
            &original.flow,
            original.evidence.iteration_residuals.clone(),
        )?;
        if evidence.structural_equilibrium_relative_residual > 10.0 * self.linear.relative_tolerance
            || evidence.interface_displacement_relative_residual
                > self.input.policy.coupling_relative_tolerance
        {
            return Err(FluidStructureError::Invalid(
                "retained structure or interface does not satisfy the coupled equations",
            ));
        }
        if evidence != original.evidence {
            return Err(FluidStructureError::Invalid(
                "retained fluid–structure evidence substitution",
            ));
        }
        Ok(())
    }

    fn characteristic_displacement(&self, displacement: &[[f64; 3]]) -> f64 {
        displacement
            .iter()
            .flat_map(|v| v.iter())
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(f64::MIN_POSITIVE)
    }

    /// Node displacement for a structural state: the structure's own nodes carry it, rigid
    /// fluid walls stay in place and the fluid interior follows through the mesh motion.
    fn configuration(&self, state: &[f64]) -> Result<Vec<[f64; 3]>, FluidStructureError> {
        let p = self.input.projection;
        let mut prescribed = vec![[0.0; 3]; p.nodes.len()];
        for (ordinal, &node) in self.structure.nodes.iter().enumerate() {
            prescribed[node] = [
                state[3 * ordinal],
                state[3 * ordinal + 1],
                state[3 * ordinal + 2],
            ];
        }
        for &node in &self.rigid_wall_nodes {
            prescribed[node] = [0.0; 3];
        }
        self.motion.extend(&prescribed, self.linear)
    }

    fn deformed_mesh(
        &self,
        displacement: &[[f64; 3]],
    ) -> Result<SolverMeshArtifact, FluidStructureError> {
        let mut mesh = self.input.mesh.clone();
        let p = self.input.projection;
        for (node, projected) in mesh.topology.nodes.iter_mut().zip(&p.nodes) {
            if node.node_id != projected.node_id {
                return Err(FluidStructureError::Invalid("projection node order"));
            }
        }
        for (node, d) in mesh.topology.nodes.iter_mut().zip(displacement) {
            let current = node.coordinates_m.canonical_value();
            node.coordinates_m =
                runmat_types::physical::Quantity::new(std::array::from_fn(|k| current[k] + d[k]));
        }
        mesh.seal_canonical_digest()
            .map_err(|e| FluidStructureError::InvalidDetail(e.to_string()))?;
        Ok(mesh)
    }

    fn fluid_topology(
        &self,
        projection: &AnalysisProjection,
    ) -> Result<FiniteVolumeTopology, FluidStructureError> {
        let topology = FiniteVolumeTopology::derive_selected(
            projection,
            self.input.geometry_policy,
            self.input.fluid_selection.clone(),
        )
        .map_err(|e| FluidStructureError::InvalidDetail(e.to_string()))?;
        if topology.cells.len() != self.reference_fluid.cells.len()
            || topology.faces.len() != self.reference_fluid.faces.len()
            || topology
                .faces
                .iter()
                .zip(&self.reference_fluid.faces)
                .any(|(a, b)| {
                    a.corner_identities != b.corner_identities
                        || a.owner != b.owner
                        || a.neighbor != b.neighbor
                })
        {
            return Err(FluidStructureError::Invalid(
                "the deformed fluid domain changed its cell or face topology",
            ));
        }
        Ok(topology)
    }

    fn flow_problem<'b>(
        &'b self,
        projection: &'b AnalysisProjection,
        topology: &'b FiniteVolumeTopology,
    ) -> Result<CanonicalFluidProblem<'b>, FluidStructureError> {
        let f = &self.input.fluid;
        Ok(CanonicalFluidProblem::prepare(CanonicalFluidInput {
            source: projection,
            topology,
            material: f.material,
            constraints: f.constraints,
            boundary_operators: f.boundary_operators,
            signals: f.signals,
            excitations: f.excitations,
            policy: f.policy,
        })?)
    }

    /// Nodal force of the fluid on the structure: each interface face's pressure and viscous
    /// traction, negated, shared by the face's consistent shape-function weights.
    fn interface_load(
        &self,
        topology: &FiniteVolumeTopology,
        flow: &CanonicalFluidResult,
    ) -> Result<Vec<f64>, FluidStructureError> {
        let p = self.input.projection;
        let forces = &flow.residual.face_forces;
        let mut load = vec![0.0; self.structure.dofs()];
        let mut seen = 0;
        for (i, face) in topology.faces.iter().enumerate() {
            let Some(source) = face.boundary_source else {
                continue;
            };
            let index = source.projection_boundary_index;
            if self.interface_faces.binary_search(&index).is_err() {
                continue;
            }
            seen += 1;
            let on_fluid = add3(
                forces.pressure_traction_n.canonical_values()[i],
                forces.viscous_traction_n.canonical_values()[i],
            );
            let f = &p.boundary_faces[index];
            let mut weights = vec![0.0; f.node_indices.len()];
            let mut area = 0.0;
            p.visit_boundary_quadrature(f, |point| {
                for (w, s) in weights.iter_mut().zip(point.shape_values()) {
                    *w += s * point.area_weight_m2;
                }
                area += point.area_weight_m2;
            })
            .map_err(|e| FluidStructureError::InvalidDetail(e.to_string()))?;
            for (&node, w) in f.node_indices.iter().zip(&weights) {
                let ordinal = self.structure.node_ordinal[node].ok_or(
                    FluidStructureError::Invalid("interface node off the structure"),
                )?;
                for k in 0..3 {
                    load[3 * ordinal + k] -= on_fluid[k] * w / area;
                }
            }
        }
        if seen != self.interface_faces.len() {
            return Err(FluidStructureError::Invalid(
                "every interface face must bound the fluid cell complex",
            ));
        }
        Ok(load)
    }

    fn interface_norm(&self, state: &[f64]) -> f64 {
        self.interface_nodes
            .iter()
            .filter_map(|&n| self.structure.node_ordinal[n])
            .flat_map(|o| state[3 * o..3 * o + 3].iter())
            .fold(0.0f64, |acc, v| acc.hypot(*v))
    }

    fn evidence(
        &self,
        displacement: &[[f64; 3]],
        response: &[f64],
        load: &[f64],
        flow: &CanonicalFluidResult,
        iteration_residuals: Vec<f64>,
    ) -> Result<FluidStructureEvidence, FluidStructureError> {
        let _ = flow;
        let retained = self
            .structure
            .nodes
            .iter()
            .flat_map(|&node| displacement[node].iter().copied())
            .collect::<Vec<_>>();
        let difference = response
            .iter()
            .zip(&retained)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let scale = self.interface_norm(response);
        let interface_displacement_relative_residual =
            relative(self.interface_norm(&difference), scale);
        let mut force = [0.0; 3];
        for chunk in load.chunks(3) {
            for k in 0..3 {
                force[k] += chunk[k];
            }
        }
        let work = dot(load, response);
        let energy = self.structure.strain_energy(response)?;
        let mut free = vec![[0.0; 3]; displacement.len()];
        for (node, d) in displacement.iter().enumerate() {
            free[node] = *d;
        }
        Ok(FluidStructureEvidence {
            iteration_residuals,
            interface_displacement_relative_residual,
            interface_force_n: force,
            interface_work_j: work,
            structural_strain_energy_j: energy,
            work_energy_relative_residual: relative((work - 2.0 * energy).abs(), work.abs()),
            structural_equilibrium_relative_residual: self
                .structure
                .equilibrium_residual(response, load)?,
            mesh_motion_relative_residual: self.motion.residual(&free)?,
            largest_interface_displacement_m: self
                .interface_nodes
                .iter()
                .map(|&n| norm3(displacement[n]))
                .fold(0.0, f64::max),
        })
    }
}

/// Conformal fluid–solid faces by role, in reference projection boundary-face order.
struct FluidSolidFaces {
    /// Faces of the declared fluid–structure interfaces: coupled.
    coupled: Vec<usize>,
    /// Faces with a fluid wall condition and no interface: rigid, uncoupled walls.
    rigid: Vec<usize>,
}

/// Classifies every conformal face joining the fluid and the structure with the shared
/// admission rule ([`crate::admission::classify_fluid_solid_face`]), which `runmat check`
/// applies to the CAD's shared faces. Each declared interface must select only fluid–solid
/// faces, and at least one.
fn fluid_solid_faces(
    model: &AnalysisModel,
    projection: &AnalysisProjection,
    fluid: &FiniteVolumeTopology,
    structure: &structure::ElasticStructure,
) -> Result<FluidSolidFaces, FluidStructureError> {
    let mut fluid_element = vec![false; projection.elements.len()];
    for cell in &fluid.cells {
        fluid_element[cell.source_element_index] = true;
    }
    let mut solid_element = vec![false; projection.elements.len()];
    for &e in &structure.elements {
        solid_element[e] = true;
    }
    let selects = |target: &runmat_analysis_core::RegionTarget, index: usize| {
        projection.boundary_faces[index]
            .provenance
            .iter()
            .any(|entity| target.matches_persistent_entity(entity))
    };
    let joins = |index: usize| {
        let f = &projection.boundary_faces[index];
        f.role == BoundaryFaceRole::ConformalInterface
            && f.adjacent_element_indices.iter().any(|&e| fluid_element[e])
            && f.adjacent_element_indices.iter().any(|&e| solid_element[e])
    };
    let mut interfaces = Vec::new();
    for interface in &model.interfaces {
        let AnalysisInterfaceKind::FluidStructure(_) = &interface.kind else {
            return Err(FluidStructureError::Unsupported(
                "non-fluid–structure interface in a fluid–structure study",
            ));
        };
        let mut selected = 0;
        for index in 0..projection.boundary_faces.len() {
            if !selects(&interface.primary, index) && !selects(&interface.secondary, index) {
                continue;
            }
            let f = &projection.boundary_faces[index];
            if f.adjacent_element_indices.len() != 2 || !joins(index) {
                return Err(FluidStructureError::InvalidDetail(format!(
                    "interface `{}` selects a face that does not join the fluid and the structure",
                    interface.interface_id
                )));
            }
            selected += 1;
        }
        if selected == 0 {
            return Err(FluidStructureError::InvalidDetail(format!(
                "interface `{}` selects no conformal fluid–structure face",
                interface.interface_id
            )));
        }
        interfaces.push(interface);
    }
    let walls = model
        .constraints
        .iter()
        .filter_map(|c| match c {
            runmat_analysis_core::AnalysisConstraint::Fluid(c)
                if crate::admission::is_fluid_wall(&c.kind) =>
            {
                fluid_wall_target(&c.kind).map(|target| (c.constraint_id.as_str(), target))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut faces = FluidSolidFaces {
        coupled: Vec::new(),
        rigid: Vec::new(),
    };
    for index in 0..projection.boundary_faces.len() {
        if !joins(index) {
            continue;
        }
        let coupling = interfaces
            .iter()
            .filter(|i| selects(&i.primary, index) || selects(&i.secondary, index))
            .map(|i| i.interface_id.as_str())
            .collect::<Vec<_>>();
        let wall = walls
            .iter()
            .filter(|(_, target)| selects(target, index))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        let face = &projection.boundary_faces[index];
        let label = format!(
            "{} at ({:.6e}, {:.6e}, {:.6e}) m",
            face.provenance
                .iter()
                .find(|e| e.kind == runmat_meshing_core::PersistentEntityKind::Face)
                .map(|e| format!("on CAD face `{}`", e.source_topology_id))
                .unwrap_or_else(|| "on the mesh".into()),
            face.centroid_m[0],
            face.centroid_m[1],
            face.centroid_m[2]
        );
        match crate::admission::classify_fluid_solid_face(&label, &coupling, &wall)? {
            crate::admission::FluidSolidFaceRole::Coupled => faces.coupled.push(index),
            crate::admission::FluidSolidFaceRole::RigidWall => faces.rigid.push(index),
        }
    }
    Ok(faces)
}

fn fluid_wall_target(kind: &FluidConstraintKind) -> Option<&runmat_analysis_core::RegionTarget> {
    match kind {
        FluidConstraintKind::NoSlipWall { target }
        | FluidConstraintKind::SlipWall { target }
        | FluidConstraintKind::MovingWall { target, .. } => Some(target),
        _ => None,
    }
}

/// A symmetric positive definite operator (structure stiffness, Newmark operator, mesh-motion
/// Laplacian) factored once by the shared sparse Cholesky and reused for every right-hand side.
pub(crate) struct SpdSystem {
    matrix: RealSparseMatrix,
    factor: std::sync::OnceLock<runmat_sparse_solvers::direct::DirectSolver<f64>>,
}

impl SpdSystem {
    pub(crate) fn new(matrix: RealSparseMatrix) -> Self {
        Self {
            matrix,
            factor: std::sync::OnceLock::new(),
        }
    }

    pub(crate) fn matrix(&self) -> &RealSparseMatrix {
        &self.matrix
    }

    fn factor(
        &self,
    ) -> Result<&runmat_sparse_solvers::direct::DirectSolver<f64>, FluidStructureError> {
        if let Some(factor) = self.factor.get() {
            return Ok(factor);
        }
        let arrays = CsrArrays::of(&self.matrix);
        let factor = with_plan(ambient_plan(), || {
            runmat_sparse_solvers::direct::DirectSolver::<f64>::new(
                arrays.view().into(),
                runmat_sparse_solvers::direct::DirectOptions::new(
                    runmat_sparse_solvers::direct::FactorKind::Cholesky,
                ),
                control(),
            )
        })
        .map_err(solver_error)?;
        Ok(self.factor.get_or_init(|| factor))
    }

    /// Solves with iterative refinement; acceptance is the residual on the original operator.
    pub(crate) fn solve(
        &self,
        b: &[f64],
        policy: LinearPolicy,
    ) -> Result<(Vec<f64>, u64), FluidStructureError> {
        if b.len() != self.matrix.rows() || b.iter().any(|v| !v.is_finite()) {
            return Err(FluidStructureError::Invalid("linear right-hand side"));
        }
        if b.iter().all(|v| *v == 0.0) {
            return Ok((vec![0.0; b.len()], 0));
        }
        let factor = self.factor()?;
        let arrays = CsrArrays::of(&self.matrix);
        let mut x = vec![0.0; b.len()];
        let report = with_plan(ambient_plan(), || {
            factor.solve_refined(arrays.view().into(), b, &mut x, 1)
        })
        .map_err(solver_error)?;
        if !(report.relative_residual <= policy.relative_tolerance)
            || x.iter().any(|v| !v.is_finite())
        {
            return Err(FluidStructureError::LinearNotConverged {
                iterations: u32::try_from(report.refinement_steps).unwrap_or(u32::MAX),
                relative_residual: Some(report.relative_residual),
            });
        }
        Ok((x, report.refinement_steps as u64))
    }
}

fn solver_error(error: runmat_sparse_solvers::SolverError) -> FluidStructureError {
    match error {
        runmat_sparse_solvers::SolverError::Cancelled => FluidStructureError::Cancelled,
        other => FluidStructureError::Operator(other.to_string()),
    }
}

pub(crate) fn relative(numerator: f64, denominator: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() {
        f64::INFINITY
    } else if denominator == 0.0 {
        if numerator == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        numerator / denominator
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn to_vectors(values: &[f64]) -> Vec<[f64; 3]> {
    values.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}
