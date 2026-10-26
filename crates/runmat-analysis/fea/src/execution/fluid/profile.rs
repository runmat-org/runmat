//! Streaming projection of the original retained U,p,q/physical-evidence wire shape.
//! No numerical arrays, solve, or replacement result is constructed to profile it.
use super::{
    invalid,
    result::{source_data, FluidResultWire},
    AnalysisExecutionResult, FluidSteadyWorkload, LIMITS,
};
use runmat_canonical_codec::{profile_serialized_shape, SerializedCanonicalProfile};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Serialize, Serializer};

struct Repeated<T> {
    sample: T,
    count: usize,
}
impl<T: Serialize> Serialize for Repeated<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(std::iter::repeat_n(&self.sample, self.count))
    }
}
fn scalar(count: usize) -> Repeated<f64> {
    Repeated { sample: 0.0, count }
}
fn vector(count: usize) -> Repeated<[f64; 3]> {
    Repeated {
        sample: [0.0; 3],
        count,
    }
}

#[derive(Serialize)]
struct FaceForcesProfile {
    #[serde(with = "runmat_types::physical::canonical_storage")]
    advective_momentum_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Repeated<[f64; 3]>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    viscous_traction_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Repeated<[f64; 3]>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pressure_traction_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Repeated<[f64; 3]>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pressure_fluctuation_traction_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Repeated<[f64; 3]>>,
}
struct BoundaryProfiles<I>(I);
impl<'a, I: Iterator<Item = &'a str> + Clone> Serialize for BoundaryProfiles<I> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.clone().map(|authority_id| {
            crate::physics::cfd::canonical::CanonicalFluidBoundaryBalance {
                authority_id: authority_id.to_owned(),
                face_count: 0,
                area_m2: runmat_types::physical::Area::new(0.0),
                volume_flow_rate_m3_per_s: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::VolumeFlowRate,
                    _,
                >::new(0.0),
                mass_flow_rate_kg_per_s: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::MassFlowRate,
                    _,
                >::new(0.0),
                advective_momentum_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new([0.0; 3]),
                viscous_traction_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new([0.0; 3]),
                pressure_traction_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new([0.0; 3]),
                pressure_fluctuation_traction_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new([0.0; 3]),
            }
        }))
    }
}

#[derive(Serialize)]
#[serde(bound(serialize = "BoundaryProfiles<I>: Serialize"))]
struct ResidualProfile<I> {
    boundary_balances: BoundaryProfiles<I>,
    face_forces: FaceForcesProfile,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    cell_vorticity_per_s: runmat_types::physical::Quantity<
        runmat_types::physical::kind::Vorticity,
        Repeated<[f64; 3]>,
    >,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    cell_momentum_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Repeated<[f64; 3]>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    cell_algebraic_momentum_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Repeated<[f64; 3]>>,
    cell_momentum_relative: Repeated<f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    cell_continuity_m3_per_s: runmat_types::physical::Quantity<
        runmat_types::physical::kind::VolumeFlowRate,
        Repeated<f64>,
    >,
    cell_continuity_relative: Repeated<f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    global_momentum_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, [f64; 3]>,
    global_momentum_relative: f64,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    global_boundary_flux_m3_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::VolumeFlowRate, f64>,
    momentum_relative: f64,
    continuity_relative: f64,
    flux_constitutive_relative: f64,
}
#[derive(Serialize)]
#[serde(bound(serialize = "ResidualProfile<I>: Serialize"))]
struct OriginalProfile<I> {
    numerical_input_identity: Digest,
    authored_authority_identity: Digest,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    velocity_m_per_s: runmat_types::physical::Quantity<
        runmat_types::physical::kind::Velocity,
        Repeated<[f64; 3]>,
    >,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pressure_fluctuation_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, Repeated<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pressure_reference_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, Repeated<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pressure_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, Repeated<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    face_volume_flux_m3_per_s: runmat_types::physical::Quantity<
        runmat_types::physical::kind::VolumeFlowRate,
        Repeated<f64>,
    >,
    outer_iterations: usize,
    linear_iterations: u64,
    residual: ResidualProfile<I>,
}
fn original_profile<I>(cells: usize, faces: usize, authorities: I) -> OriginalProfile<I> {
    OriginalProfile {
        numerical_input_identity: Digest::from_bytes([0; 32]),
        authored_authority_identity: Digest::from_bytes([0; 32]),
        velocity_m_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Velocity,
            _,
        >::new(vector(cells)),
        pressure_fluctuation_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(scalar(cells)),
        pressure_reference_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(scalar(cells)),
        pressure_pa:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Pressure, _>::new(
                scalar(cells),
            ),
        face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::VolumeFlowRate,
            _,
        >::new(scalar(faces)),
        outer_iterations: 0,
        linear_iterations: 0,
        residual: ResidualProfile {
            boundary_balances: BoundaryProfiles(authorities),
            face_forces: FaceForcesProfile {
                advective_momentum_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new(vector(faces)),
                viscous_traction_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new(vector(faces)),
                pressure_traction_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new(vector(faces)),
                pressure_fluctuation_traction_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new(vector(faces)),
            },
            cell_vorticity_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Vorticity,
                _,
            >::new(vector(cells)),
            cell_momentum_n: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Force,
                _,
            >::new(vector(cells)),
            cell_algebraic_momentum_n: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Force,
                _,
            >::new(vector(cells)),
            cell_momentum_relative: scalar(cells),
            cell_continuity_m3_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::VolumeFlowRate,
                _,
            >::new(scalar(cells)),
            cell_continuity_relative: scalar(cells),
            global_momentum_n: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Force,
                _,
            >::new([0.0; 3]),
            global_momentum_relative: 0.0,
            global_boundary_flux_m3_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::VolumeFlowRate,
                _,
            >::new(0.0),
            momentum_relative: 0.0,
            continuity_relative: 0.0,
            flux_constitutive_relative: 0.0,
        },
    }
}

/// Same variant names as the retained `FluidRetainedOriginal` wire enum.
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum RetainedShape<S, T, F = (), E = (), G = ()> {
    Steady(S),
    Transient(T),
    FluidStructure(G),
    FluidStructureTransient(G),
    ConjugateHeatTransient(G),
    ConjugateHeat { flow: F, energy: E },
}

/// Field order of `ConjugateHeatTransientResult`, with maximum-width placeholder values.
#[derive(Serialize)]
struct ConjugateHeatTransientShape<F> {
    flow: F,
    departure_time_s: f64,
    departure: crate::physics::conjugate_heat::ConjugateHeatResult,
    energy: crate::physics::conjugate_heat::ConjugateHeatResult,
    steps: Repeated<crate::physics::conjugate_heat::ConjugateHeatStep>,
}

/// Field order of `FluidStructureTransientResult`, with maximum-width placeholder values.
#[derive(Serialize)]
struct FluidStructureTransientShape<F> {
    start_time_s: f64,
    departure_time_s: f64,
    departure_displacement_m: Repeated<[f64; 3]>,
    departure_structure: crate::physics::fluid_structure::NewmarkState,
    departure_flow: crate::physics::fluid_structure::FluidStructureFlowState,
    departure_load: Repeated<f64>,
    displacement_m: Repeated<[f64; 3]>,
    structure: crate::physics::fluid_structure::NewmarkState,
    flow: F,
    steps: Repeated<crate::physics::fluid_structure::FluidStructureStepEvidence>,
    energy_relative_residual: f64,
}

/// Field order of `FluidStructureResult`, with maximum-width placeholder values.
#[derive(Serialize)]
struct FluidStructureShape<F> {
    displacement_m: Repeated<[f64; 3]>,
    structural_response_m: Repeated<[f64; 3]>,
    flow: F,
    coupling_iterations: usize,
    evidence: crate::physics::fluid_structure::FluidStructureEvidence,
}

pub(super) fn original_block_profile<'a>(
    cells: usize,
    faces: usize,
    authorities: impl Iterator<Item = &'a str> + Clone,
    intervals: usize,
    retained: usize,
) -> impl Serialize {
    #[derive(Serialize)]
    #[serde(bound(serialize = "BoundaryProfiles<I>: Serialize"))]
    struct Block<I> {
        identity: Digest,
        start_ordinal: usize,
        end_ordinal: usize,
        intervals: Repeated<super::FluidIntervalEvidence>,
        retained: Repeated<(usize, OriginalProfile<I>)>,
    }
    Block {
        identity: Digest::from_bytes([0; 32]),
        start_ordinal: usize::MAX,
        end_ordinal: usize::MAX,
        intervals: Repeated {
            sample: super::FluidIntervalEvidence {
                ordinal: usize::MAX,
                outer_iterations: usize::MAX,
                linear_iterations: u64::MAX,
                momentum_relative: f64::MAX,
                continuity_relative: f64::MAX,
                fields_digest: Digest::from_bytes([0; 32]),
            },
            count: intervals,
        },
        retained: Repeated {
            sample: (usize::MAX, original_profile(cells, faces, authorities)),
            count: retained,
        },
    }
}

pub(super) fn original_block_encoding_profile<'a>(
    cells: usize,
    faces: usize,
    authorities: impl Iterator<Item = &'a str> + Clone,
    intervals: usize,
    retained: usize,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    Ok(profile_serialized_shape(
        &original_block_profile(cells, faces, authorities, intervals, retained),
        LIMITS,
    )?)
}

impl FluidSteadyWorkload {
    /// Maximum-width shared codec profile of the actual immutable result schema.
    /// Exact source topology determines cell/face cardinality; numeric samples
    /// reserve all finite values and do not stand in for physical solve evidence.
    pub fn result_encoding_profile(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        let source = self.prepare(mesh)?;
        let problem = source.problem(self)?;
        self.result_encoding_profile_prepared(mesh, &source, problem)
    }

    /// [`Self::result_encoding_profile`] over a source and problem the caller already prepared
    /// from this workload and mesh, so resource derivation prepares the original support once.
    pub(crate) fn result_encoding_profile_prepared(
        &self,
        mesh: &SolverMeshArtifact,
        source: &super::source::PreparedFluidSource,
        problem: crate::physics::cfd::canonical::CanonicalFluidProblem<'_>,
    ) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        let reservation = problem.reservation();
        let identity = Digest::from_bytes([0; 32]);
        if self.coupling() == super::workload::FluidCoupling::FluidStructure && self.is_transient()
        {
            let coupled = source.fluid_structure(self, mesh)?;
            let schedule =
                super::FluidTransientSchedule::derive(&self.model, &self.model.sequence.steps[0])?;
            let cells = source.topology.cells.len();
            let faces = source.topology.faces.len();
            let structural = 3 * coupled.structural_node_count();
            let flow = original_profile(cells, faces, problem.boundary_balance_authorities());
            let state = || crate::physics::fluid_structure::NewmarkState {
                displacement: vec![0.0; structural],
                velocity: vec![0.0; structural],
                acceleration: vec![0.0; structural],
            };
            let shape =
                RetainedShape::<(), (), (), (), _>::FluidStructureTransient(
                    FluidStructureTransientShape {
                        start_time_s: 0.0,
                        departure_time_s: 0.0,
                        departure_displacement_m: vector(source.source.nodes.len()),
                        departure_structure: state(),
                        departure_flow: crate::physics::fluid_structure::FluidStructureFlowState {
                            velocity_m_per_s: runmat_types::physical::Quantity::new(vec![
                                [0.0; 3];
                                cells
                            ]),
                            pressure_reference_pa: runmat_types::physical::Quantity::new(vec![
                                0.0;
                                cells
                            ]),
                            pressure_fluctuation_pa: runmat_types::physical::Quantity::new(
                                vec![0.0; cells],
                            ),
                            face_volume_flux_m3_per_s: runmat_types::physical::Quantity::new(
                                vec![0.0; faces],
                            ),
                        },
                        departure_load: scalar(structural),
                        displacement_m: vector(source.source.nodes.len()),
                        structure: state(),
                        flow,
                        steps: Repeated {
                            sample: crate::physics::fluid_structure::FluidStructureStepEvidence {
                                time_s: 0.0,
                                coupling_iterations: 0,
                                interface_displacement_relative_residual: 0.0,
                                interface_force_n: [0.0; 3],
                                interface_work_j: 0.0,
                                strain_energy_j: 0.0,
                                kinetic_energy_j: 0.0,
                                structural_dynamic_relative_residual: 0.0,
                            },
                            count: schedule.points_s().len() - 1,
                        },
                        energy_relative_residual: 0.0,
                    },
                );
            let data = source_data(self, source, reservation, identity, shape)?;
            return profile_serialized_shape(&FluidResultWire { data }, LIMITS).map_err(invalid);
        }
        if self.is_transient() && self.coupling() == super::workload::FluidCoupling::Standalone {
            drop(problem);
            let view = self.transient_view();
            let prepared = view.prepare(mesh)?;
            let shape = RetainedShape::<(), _>::Transient(prepared.result_shape()?);
            let data = source_data(self, source, reservation, identity, shape)?;
            return profile_serialized_shape(&FluidResultWire { data }, LIMITS).map_err(invalid);
        }
        let original = original_profile(
            source.topology.cells.len(),
            source.topology.faces.len(),
            problem.boundary_balance_authorities(),
        );
        if self.coupling() == super::workload::FluidCoupling::FluidStructure {
            let coupled = source.fluid_structure(self, mesh)?;
            let shape = RetainedShape::<(), (), (), (), _>::FluidStructure(FluidStructureShape {
                displacement_m: vector(source.source.nodes.len()),
                structural_response_m: vector(coupled.structural_node_count()),
                flow: original,
                coupling_iterations: 0,
                evidence: crate::physics::fluid_structure::FluidStructureEvidence {
                    iteration_residuals: vec![0.0; coupled.maximum_coupling_iterations()],
                    interface_displacement_relative_residual: 0.0,
                    interface_force_n: [0.0; 3],
                    interface_work_j: 0.0,
                    structural_strain_energy_j: 0.0,
                    work_energy_relative_residual: 0.0,
                    structural_equilibrium_relative_residual: 0.0,
                    mesh_motion_relative_residual: 0.0,
                    largest_interface_displacement_m: 0.0,
                },
            });
            let data = source_data(self, source, reservation, identity, shape)?;
            return profile_serialized_shape(&FluidResultWire { data }, LIMITS).map_err(invalid);
        }
        if self.coupling() == super::workload::FluidCoupling::ConjugateHeat && self.is_transient() {
            let flux = vec![0.0; source.topology.faces.len()];
            let energy = source.conjugate_heat(self, &flux)?.result_shape();
            let schedule =
                super::FluidTransientSchedule::derive(&self.model, &self.model.sequence.steps[0])?;
            let shape = RetainedShape::<(), (), (), (), _>::ConjugateHeatTransient(
                ConjugateHeatTransientShape {
                    flow: original,
                    departure_time_s: 0.0,
                    departure: energy.clone(),
                    energy,
                    steps: Repeated {
                        sample: crate::physics::conjugate_heat::ConjugateHeatStep {
                            time_s: 0.0,
                            peak_temperature_k: 0.0,
                            stored_heat_rate_w: 0.0,
                            relative_imbalance: 0.0,
                        },
                        count: schedule.points_s().len() - 1,
                    },
                },
            );
            let data = source_data(self, source, reservation, identity, shape)?;
            return profile_serialized_shape(&FluidResultWire { data }, LIMITS).map_err(invalid);
        }
        if self.coupling() == super::workload::FluidCoupling::ConjugateHeat {
            // Zero flux fixes every cardinality of the energy result without a solve.
            let flux = vec![0.0; source.topology.faces.len()];
            let energy = source.conjugate_heat(self, &flux)?.result_shape();
            let shape = RetainedShape::<(), (), _, _, ()>::ConjugateHeat {
                flow: original,
                energy,
            };
            let data = source_data(self, source, reservation, identity, shape)?;
            return profile_serialized_shape(&FluidResultWire { data }, LIMITS).map_err(invalid);
        }
        let shape = RetainedShape::<_, ()>::Steady(original);
        let data = source_data(self, source, reservation, identity, shape)?;
        let wire = FluidResultWire { data };
        profile_serialized_shape(&wire, LIMITS).map_err(invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::cfd::canonical::{
        CanonicalFluidFaceForces, CanonicalFluidResidual, CanonicalFluidResult,
    };

    #[test]
    fn streaming_numerical_shape_matches_actual_serializer_for_arbitrary_finite_values() {
        for (cells, faces) in [(0, 0), (1, 4), (8, 23)] {
            let v = [f64::MAX, -f64::MAX, f64::MIN_POSITIVE];
            let original = CanonicalFluidResult {
                numerical_input_identity: Digest::from_bytes([1; 32]),
                authored_authority_identity: Digest::from_bytes([2; 32]),
                velocity_m_per_s: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Velocity,
                    _,
                >::new(vec![v; cells]),
                pressure_fluctuation_pa: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Pressure,
                    _,
                >::new(vec![f64::MAX; cells]),
                pressure_reference_pa: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Pressure,
                    _,
                >::new(vec![0.0; cells]),
                pressure_pa: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Pressure,
                    _,
                >::new(vec![f64::MAX; cells]),
                face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::VolumeFlowRate,
                    _,
                >::new(vec![-f64::MAX; faces]),
                outer_iterations: 123,
                linear_iterations: 1234567,
                residual: CanonicalFluidResidual {
                    cell_vorticity_per_s: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Vorticity,
                        _,
                    >::new(vec![[0.0; 3]; cells]),
                    boundary_balances: vec![
                            crate::physics::cfd::canonical::CanonicalFluidBoundaryBalance {
                                authority_id: "pressure_β".into(),
                                face_count: u64::MAX,
                                area_m2: runmat_types::physical::Quantity::<
                                    runmat_types::physical::kind::Area,
                                    _,
                                >::new(f64::MAX),
                                volume_flow_rate_m3_per_s: runmat_types::physical::Quantity::<
                                    runmat_types::physical::kind::VolumeFlowRate,
                                    _,
                                >::new(
                                    -f64::MAX
                                ),
                                mass_flow_rate_kg_per_s: runmat_types::physical::Quantity::<
                                    runmat_types::physical::kind::MassFlowRate,
                                    _,
                                >::new(
                                    f64::MIN_POSITIVE
                                ),
                                advective_momentum_n: runmat_types::physical::Quantity::<
                                    runmat_types::physical::kind::Force,
                                    _,
                                >::new(v),
                                viscous_traction_n: runmat_types::physical::Quantity::<
                                    runmat_types::physical::kind::Force,
                                    _,
                                >::new(v),
                                pressure_traction_n: runmat_types::physical::Quantity::<
                                    runmat_types::physical::kind::Force,
                                    _,
                                >::new(v),
                                pressure_fluctuation_traction_n: runmat_types::physical::Quantity::<
                                    runmat_types::physical::kind::Force,
                                    _,
                                >::new(
                                    v
                                ),
                            };
                            faces
                        ],
                    face_forces: CanonicalFluidFaceForces {
                        advective_momentum_n: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Force,
                            _,
                        >::new(vec![v; faces]),
                        viscous_traction_n: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Force,
                            _,
                        >::new(vec![v; faces]),
                        pressure_traction_n: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Force,
                            _,
                        >::new(vec![v; faces]),
                        pressure_fluctuation_traction_n: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Force,
                            _,
                        >::new(vec![
                            v;
                            faces
                        ]),
                    },
                    cell_momentum_n: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Force,
                        _,
                    >::new(vec![v; cells]),
                    cell_algebraic_momentum_n: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Force,
                        _,
                    >::new(vec![v; cells]),
                    cell_momentum_relative: vec![f64::MAX; cells],
                    cell_continuity_m3_per_s: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::VolumeFlowRate,
                        _,
                    >::new(vec![-f64::MAX; cells]),
                    cell_continuity_relative: vec![f64::MIN_POSITIVE; cells],
                    global_momentum_n: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Force,
                        _,
                    >::new(v),
                    global_momentum_relative: f64::MAX,
                    global_boundary_flux_m3_per_s: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::VolumeFlowRate,
                        _,
                    >::new(-f64::MAX),
                    momentum_relative: f64::MIN_POSITIVE,
                    continuity_relative: f64::MAX,
                    flux_constitutive_relative: -f64::MAX,
                },
            };
            // The projection must have exactly the shape of a real result.
            assert_eq!(
                profile_serialized_shape(&original, LIMITS).unwrap(),
                profile_serialized_shape(
                    &original_profile(cells, faces, std::iter::repeat_n("pressure_β", faces)),
                    LIMITS
                )
                .unwrap(),
            );
        }
    }
}

impl<T: runmat_types::physical::QuantityStorage> runmat_types::physical::QuantityStorage
    for Repeated<T>
{
    fn all_finite(&self) -> bool {
        self.sample.all_finite()
    }
}
