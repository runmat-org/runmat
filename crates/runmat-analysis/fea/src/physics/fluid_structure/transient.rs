//! Transient fluid–structure interaction: a moving-mesh backward-Euler flow interval and an
//! average-acceleration Newmark structure, iterated within every interval until the
//! structure's response to the flow reproduces the wall motion that shaped it.
//!
//! The fluid mesh sweeps exact face volumes between configurations (geometric conservation),
//! the wetted wall moves with the structure's velocity and carries exactly its swept flux, and
//! the structure is loaded by the negated pressure and viscous traction of every wetted face.
//! Average acceleration conserves energy for a linear structure: the work of the interface
//! traction over each interval equals the change of strain plus kinetic energy, which every
//! accepted interval reports with its coupling residuals.
use super::{structure::NewmarkState, FluidStructureError as Error, FluidStructureProblem};
use crate::{
    excitation::ProjectionSample,
    physics::cfd::canonical::{
        swept_face_flux, CanonicalFluidFields, CanonicalFluidMeshMotion, CanonicalFluidProblem,
        CanonicalFluidResult,
    },
    topology::{AnalysisProjection, FiniteVolumeTopology},
};
use serde::{Deserialize, Serialize};

/// Original flow fields of one instant in reference cell and face order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidStructureFlowState {
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub velocity_m_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Velocity, Vec<[f64; 3]>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub pressure_reference_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, Vec<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub pressure_fluctuation_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, Vec<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub face_volume_flux_m3_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::VolumeFlowRate, Vec<f64>>,
}

impl FluidStructureFlowState {
    fn of(result: &CanonicalFluidResult) -> Self {
        Self {
            velocity_m_per_s: result.velocity_m_per_s.clone(),
            pressure_reference_pa: result.pressure_reference_pa.clone(),
            pressure_fluctuation_pa: result.pressure_fluctuation_pa.clone(),
            face_volume_flux_m3_per_s: result.face_volume_flux_m3_per_s.clone(),
        }
    }
    fn fields(&self) -> CanonicalFluidFields<'_> {
        CanonicalFluidFields {
            velocity_m_per_s: self.velocity_m_per_s.as_quantity_slice(),
            pressure_reference_pa: self.pressure_reference_pa.as_quantity_slice(),
            pressure_fluctuation_pa: self.pressure_fluctuation_pa.as_quantity_slice(),
            face_volume_flux_m3_per_s: self.face_volume_flux_m3_per_s.as_quantity_slice(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidStructureStepEvidence {
    pub time_s: f64,
    pub coupling_iterations: usize,
    pub interface_displacement_relative_residual: f64,
    pub interface_force_n: [f64; 3],
    /// Work of the interval-mean interface traction on the interval displacement, J.
    pub interface_work_j: f64,
    pub strain_energy_j: f64,
    pub kinetic_energy_j: f64,
    pub structural_dynamic_relative_residual: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidStructureTransientResult {
    pub start_time_s: f64,
    /// State departing the final interval, so that interval's equations recover cold.
    pub departure_time_s: f64,
    pub departure_displacement_m: Vec<[f64; 3]>,
    pub departure_structure: NewmarkState,
    pub departure_flow: FluidStructureFlowState,
    pub departure_load: Vec<f64>,
    /// Final instant: node displacement shaping the flow, structure state and flow.
    pub displacement_m: Vec<[f64; 3]>,
    pub structure: NewmarkState,
    pub flow: CanonicalFluidResult,
    pub steps: Vec<FluidStructureStepEvidence>,
    /// Cumulative interface work minus the stored strain and kinetic energy, relative to it.
    pub energy_relative_residual: f64,
}

struct Configuration {
    displacement: Vec<[f64; 3]>,
    projection: AnalysisProjection,
    topology: FiniteVolumeTopology,
}

impl FluidStructureProblem<'_> {
    fn configure(&self, state: &[f64]) -> Result<Configuration, Error> {
        let displacement = self.configuration(state)?;
        let mesh = self.deformed_mesh(&displacement)?;
        let projection =
            AnalysisProjection::derive(&mesh).map_err(|e| Error::InvalidDetail(e.to_string()))?;
        let topology = self.fluid_topology(&projection)?;
        Ok(Configuration {
            displacement,
            projection,
            topology,
        })
    }

    /// Wetted-wall velocity of each interface face: the face-mean structural velocity whose
    /// normal component is corrected to carry exactly the face's swept flux.
    fn wall_velocities(
        &self,
        topology: &FiniteVolumeTopology,
        velocity: &[f64],
        mesh_flux: &[f64],
    ) -> Result<Vec<(usize, [f64; 3])>, Error> {
        let p = self.input.projection;
        let mut walls = Vec::new();
        for (i, face) in topology.faces.iter().enumerate() {
            let Some(source) = face.boundary_source else {
                continue;
            };
            if self
                .interface_faces
                .binary_search(&source.projection_boundary_index)
                .is_err()
            {
                continue;
            }
            let nodes = &p.boundary_faces[source.projection_boundary_index].node_indices;
            let mut mean = [0.0; 3];
            for &node in nodes {
                let o = self.structure.node_ordinal[node]
                    .ok_or(Error::Invalid("interface node off the structure"))?;
                for k in 0..3 {
                    mean[k] += velocity[3 * o + k] / nodes.len() as f64;
                }
            }
            let s = face.owner_area_vector_m2;
            let area2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
            let excess =
                (mesh_flux[i] - (mean[0] * s[0] + mean[1] * s[1] + mean[2] * s[2])) / area2;
            walls.push((i, std::array::from_fn(|k| mean[k] + excess * s[k])));
        }
        Ok(walls)
    }

    fn interval<'b>(
        &'b self,
        arrival: &'b Configuration,
        time_s: f64,
        walls: &[(usize, [f64; 3])],
    ) -> Result<CanonicalFluidProblem<'b>, Error> {
        let f = &self.input.fluid;
        Ok(CanonicalFluidProblem::prepare_at(
            crate::physics::cfd::canonical::CanonicalFluidInput {
                source: &arrival.projection,
                topology: &arrival.topology,
                material: f.material,
                constraints: f.constraints,
                boundary_operators: f.boundary_operators,
                signals: f.signals,
                excitations: f.excitations,
                policy: f.policy,
            },
            ProjectionSample::TimeSeconds {
                value: time_s,
                approach: runmat_analysis_core::SignalSampleApproach::At,
            },
        )?
        .with_wall_velocities(walls)?)
    }

    /// March the coupled system over `times` from rest at `times[0]`.
    pub fn solve_transient(&self, times: &[f64]) -> Result<FluidStructureTransientResult, Error> {
        if times.len() < 2 || times.windows(2).any(|w| !(w[1] > w[0])) {
            return Err(Error::Invalid("transient fluid–structure time grid"));
        }
        let dofs = self.structure.dofs();
        let mut structure = NewmarkState::rest(dofs);
        let mut departure = self.configure(&structure.displacement)?;
        let cells = departure.topology.cells.len();
        let mut flow = FluidStructureFlowState {
            velocity_m_per_s: runmat_types::physical::Quantity::new(vec![[0.0; 3]; cells]),
            pressure_reference_pa: runmat_types::physical::Quantity::new(Vec::new()),
            pressure_fluctuation_pa: runmat_types::physical::Quantity::new(vec![0.0; cells]),
            face_volume_flux_m3_per_s: runmat_types::physical::Quantity::new(vec![
                0.0;
                departure
                    .topology
                    .faces
                    .len()
            ]),
        };
        let mut load = vec![0.0; dofs];
        let mut steps = Vec::new();
        let mut work = 0.0;
        let mut operator: Option<(f64, super::SpdSystem)> = None;
        for (step_index, window) in times.windows(2).enumerate() {
            let started = runmat_time::Instant::now();
            let (t0, t1) = (window[0], window[1]);
            let dt = t1 - t0;
            if operator.as_ref().is_none_or(|(step, _)| *step != dt) {
                operator = Some((dt, self.structure.newmark_operator(dt)?));
            }
            let newmark = &operator.as_ref().unwrap().1;
            let mut guess = structure
                .displacement
                .iter()
                .zip(&structure.velocity)
                .map(|(u, v)| u + dt * v)
                .collect::<Vec<_>>();
            let mut previous_residual: Option<Vec<f64>> = None;
            let mut relaxation = self.input.policy.initial_relaxation;
            let mut residuals = Vec::new();
            let mut accepted = None;
            // The fluid solve of each coupling iteration after the first starts from the
            // previous iteration's flow: the interface moves little between them.
            let mut previous_flow: Option<crate::physics::cfd::canonical::CanonicalFluidResult> =
                None;
            for iteration in 1..=self.input.policy.maximum_coupling_iterations {
                if crate::is_fea_cancelled() {
                    return Err(Error::Cancelled);
                }
                let kinematics = structure.advance(dt, guess.clone());
                let arrival = self.configure(&guess)?;
                let mesh_flux = swept_face_flux(
                    &departure.projection,
                    &arrival.projection,
                    &arrival.topology,
                    dt,
                )?;
                let walls =
                    self.wall_velocities(&arrival.topology, &kinematics.velocity, &mesh_flux)?;
                let problem = self.interval(&arrival, t1, &walls)?;
                if flow.pressure_reference_pa.canonical_values().is_empty() {
                    flow.pressure_reference_pa = runmat_types::physical::Quantity::new(
                        problem.pressure_reference().to_vec(),
                    );
                }
                let previous_volume = departure
                    .topology
                    .cells
                    .iter()
                    .map(|c| c.volume_m3)
                    .collect::<Vec<_>>();
                let step = problem.prepare_backward_euler_moving(
                    t0,
                    t1,
                    flow.fields(),
                    CanonicalFluidMeshMotion {
                        previous_volume_m3: &previous_volume,
                        face_mesh_flux_m3_per_s: &mesh_flux,
                    },
                )?;
                let result = match &previous_flow {
                    Some(previous) => step.solve_from(previous.original_fields())?,
                    None => step.solve()?,
                };
                previous_flow = Some(result.clone());
                let arrival_load = self.interface_load(&arrival.topology, &result)?;
                let response = self.structure.newmark_step(
                    newmark,
                    dt,
                    &structure,
                    &arrival_load,
                    self.linear,
                )?;
                let residual = response
                    .displacement
                    .iter()
                    .zip(&guess)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>();
                let scale = self.interface_norm(&response.displacement);
                let relative = if scale == 0.0 && self.interface_norm(&residual) == 0.0 {
                    0.0
                } else {
                    self.interface_norm(&residual) / scale.max(f64::MIN_POSITIVE)
                };
                residuals.push(relative);
                if relative <= self.input.policy.coupling_relative_tolerance {
                    accepted = Some((
                        iteration,
                        kinematics,
                        arrival,
                        result,
                        arrival_load,
                        response,
                    ));
                    break;
                }
                if let Some(previous) = &previous_residual {
                    let delta = residual
                        .iter()
                        .zip(previous)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>();
                    let denominator = super::dot(&delta, &delta);
                    if denominator > 0.0 {
                        relaxation = (-relaxation * super::dot(previous, &delta) / denominator)
                            .clamp(-1.0, 1.0);
                        if relaxation.abs() < 1e-3 {
                            relaxation = 1e-3f64.copysign(relaxation);
                        }
                    }
                }
                for (g, r) in guess.iter_mut().zip(&residual) {
                    *g += relaxation * r;
                }
                previous_residual = Some(residual);
            }
            let Some((iterations, kinematics, arrival, result, arrival_load, response)) = accepted
            else {
                return Err(Error::CouplingNotConverged {
                    iterations: self.input.policy.maximum_coupling_iterations,
                    residuals,
                });
            };
            let increment = kinematics
                .displacement
                .iter()
                .zip(&structure.displacement)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            let mean_load = load
                .iter()
                .zip(&arrival_load)
                .map(|(a, b)| 0.5 * (a + b))
                .collect::<Vec<_>>();
            work += super::dot(&mean_load, &increment);
            let strain = self.structure.strain_energy(&kinematics.displacement)?;
            let kinetic = self.structure.kinetic_energy(&kinematics.velocity)?;
            let mut force = [0.0; 3];
            for chunk in arrival_load.chunks(3) {
                for k in 0..3 {
                    force[k] += chunk[k];
                }
            }
            steps.push(FluidStructureStepEvidence {
                time_s: t1,
                coupling_iterations: iterations,
                interface_displacement_relative_residual: *residuals.last().unwrap(),
                interface_force_n: force,
                interface_work_j: work,
                strain_energy_j: strain,
                kinetic_energy_j: kinetic,
                structural_dynamic_relative_residual: self
                    .structure
                    .dynamic_residual(&response, &arrival_load)?,
            });
            crate::progress::report_step(
                "fea.run_fluid_structure_transient",
                step_index + 1,
                times.len() - 1,
                t1,
                started,
                format_args!(
                    " ({iterations} coupling iterations, last fluid solve {} outer and {} linear iterations)",
                    result.outer_iterations, result.linear_iterations
                ),
            );
            let is_last = times.last() == Some(&t1);
            let mut last = is_last.then(|| {
                (
                    t0,
                    departure.displacement.clone(),
                    structure.clone(),
                    flow.clone(),
                    load.clone(),
                )
            });
            structure = kinematics;
            flow = FluidStructureFlowState::of(&result);
            load = arrival_load;
            departure = arrival;
            if is_last {
                let (
                    departure_time_s,
                    departure_displacement_m,
                    departure_structure,
                    departure_flow,
                    departure_load,
                ) = last.take().expect("one interval");
                let stored = steps
                    .last()
                    .map_or(0.0, |s| s.strain_energy_j + s.kinetic_energy_j);
                return Ok(FluidStructureTransientResult {
                    start_time_s: times[0],
                    departure_time_s,
                    departure_displacement_m,
                    departure_structure,
                    departure_flow,
                    departure_load,
                    displacement_m: departure.displacement,
                    structure,
                    flow: result,
                    steps,
                    energy_relative_residual: super::relative(
                        (work - stored).abs(),
                        work.abs().max(stored),
                    ),
                });
            }
        }
        Err(Error::Invalid("transient fluid–structure time grid"))
    }

    /// Re-evaluate the final interval from the retained departure state: the moving-mesh flow
    /// equations, the structure's dynamic equilibrium and the interface agreement.
    pub fn recover_transient(&self, original: &FluidStructureTransientResult) -> Result<(), Error> {
        let dt = original
            .steps
            .last()
            .map(|s| s.time_s - original.departure_time_s)
            .ok_or(Error::Invalid("retained transient fluid–structure history"))?;
        let t1 = original.departure_time_s + dt;
        let state = |d: &[[f64; 3]]| -> Vec<f64> {
            self.structure
                .nodes
                .iter()
                .flat_map(|&n| d[n].iter().copied())
                .collect()
        };
        if state(&original.departure_displacement_m) != original.departure_structure.displacement
            || state(&original.displacement_m) != original.structure.displacement
        {
            return Err(Error::Invalid(
                "retained displacement differs from the structural state",
            ));
        }
        let departure = self.configure(&original.departure_structure.displacement)?;
        let arrival = self.configure(&original.structure.displacement)?;
        let mesh_flux = swept_face_flux(
            &departure.projection,
            &arrival.projection,
            &arrival.topology,
            dt,
        )?;
        let walls =
            self.wall_velocities(&arrival.topology, &original.structure.velocity, &mesh_flux)?;
        let problem = self.interval(&arrival, t1, &walls)?;
        let previous_volume = departure
            .topology
            .cells
            .iter()
            .map(|c| c.volume_m3)
            .collect::<Vec<_>>();
        problem
            .prepare_backward_euler_moving(
                original.departure_time_s,
                t1,
                original.departure_flow.fields(),
                CanonicalFluidMeshMotion {
                    previous_volume_m3: &previous_volume,
                    face_mesh_flux_m3_per_s: &mesh_flux,
                },
            )?
            .recover(&original.flow)?;
        let kinematics = original
            .departure_structure
            .advance(dt, original.structure.displacement.clone());
        if kinematics != original.structure {
            return Err(Error::Invalid(
                "retained structural kinematics differ from Newmark",
            ));
        }
        let load = self.interface_load(&arrival.topology, &original.flow)?;
        let residual = self
            .structure
            .dynamic_residual(&original.structure, &load)?;
        let tolerance = self
            .input
            .policy
            .coupling_relative_tolerance
            .max(10.0 * self.linear.relative_tolerance);
        if residual > 1e3 * tolerance {
            return Err(Error::Invalid(
                "retained structure is out of dynamic equilibrium with the retained flow",
            ));
        }
        Ok(())
    }
}
