//! Streaming projection of the original retained U,p,q/physical-evidence wire shape.
//! No numerical arrays, solve, or replacement result is constructed to profile it.
use super::{
    invalid,
    result::{source_data, FluidResultWire},
    AnalysisExecutionResult, FluidSteadyWorkload, LIMITS,
};
use runmat_canonical_codec::{profile_serialized, SerializedCanonicalProfile};
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
    advective_momentum_n: Repeated<[f64; 3]>,
    viscous_traction_n: Repeated<[f64; 3]>,
    pressure_traction_n: Repeated<[f64; 3]>,
    pressure_fluctuation_traction_n: Repeated<[f64; 3]>,
}
struct BoundaryProfiles<I>(I);
impl<'a, I: Iterator<Item = &'a str> + Clone> Serialize for BoundaryProfiles<I> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.clone().map(|authority_id| {
            crate::physics::cfd::canonical::CanonicalFluidBoundaryBalance {
                authority_id: authority_id.to_owned(),
                face_count: 0,
                area_m2: 0.0,
                volume_flow_rate_m3_per_s: 0.0,
                mass_flow_rate_kg_per_s: 0.0,
                advective_momentum_n: [0.0; 3],
                viscous_traction_n: [0.0; 3],
                pressure_traction_n: [0.0; 3],
                pressure_fluctuation_traction_n: [0.0; 3],
            }
        }))
    }
}

#[derive(Serialize)]
#[serde(bound(serialize = "BoundaryProfiles<I>: Serialize"))]
struct ResidualProfile<I> {
    boundary_balances: BoundaryProfiles<I>,
    face_forces: FaceForcesProfile,
    cell_momentum_n: Repeated<[f64; 3]>,
    cell_algebraic_momentum_n: Repeated<[f64; 3]>,
    cell_momentum_relative: Repeated<f64>,
    cell_continuity_m3_per_s: Repeated<f64>,
    cell_continuity_relative: Repeated<f64>,
    global_momentum_n: [f64; 3],
    global_momentum_relative: f64,
    global_boundary_flux_m3_per_s: f64,
    momentum_relative: f64,
    continuity_relative: f64,
    flux_constitutive_relative: f64,
}
#[derive(Serialize)]
#[serde(bound(serialize = "ResidualProfile<I>: Serialize"))]
struct OriginalProfile<I> {
    numerical_input_identity: Digest,
    authored_authority_identity: Digest,
    velocity_m_per_s: Repeated<[f64; 3]>,
    pressure_fluctuation_pa: Repeated<f64>,
    pressure_pa: Repeated<f64>,
    face_volume_flux_m3_per_s: Repeated<f64>,
    outer_iterations: usize,
    linear_iterations: u64,
    residual: ResidualProfile<I>,
}
fn original_profile<I>(cells: usize, faces: usize, authorities: I) -> OriginalProfile<I> {
    OriginalProfile {
        numerical_input_identity: Digest::from_bytes([0; 32]),
        authored_authority_identity: Digest::from_bytes([0; 32]),
        velocity_m_per_s: vector(cells),
        pressure_fluctuation_pa: scalar(cells),
        pressure_pa: scalar(cells),
        face_volume_flux_m3_per_s: scalar(faces),
        outer_iterations: 0,
        linear_iterations: 0,
        residual: ResidualProfile {
            boundary_balances: BoundaryProfiles(authorities),
            face_forces: FaceForcesProfile {
                advective_momentum_n: vector(faces),
                viscous_traction_n: vector(faces),
                pressure_traction_n: vector(faces),
                pressure_fluctuation_traction_n: vector(faces),
            },
            cell_momentum_n: vector(cells),
            cell_algebraic_momentum_n: vector(cells),
            cell_momentum_relative: scalar(cells),
            cell_continuity_m3_per_s: scalar(cells),
            cell_continuity_relative: scalar(cells),
            global_momentum_n: [0.0; 3],
            global_momentum_relative: 0.0,
            global_boundary_flux_m3_per_s: 0.0,
            momentum_relative: 0.0,
            continuity_relative: 0.0,
            flux_constitutive_relative: 0.0,
        },
    }
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
        let reservation = problem.reservation();
        let original = original_profile(
            source.topology.cells.len(),
            source.topology.faces.len(),
            problem.boundary_balance_authorities(),
        );
        let data = source_data(
            self,
            &source,
            reservation,
            original.numerical_input_identity,
            original,
        )?;
        let wire = FluidResultWire { data };
        profile_serialized(&wire, LIMITS).map_err(invalid)
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
                velocity_m_per_s: vec![v; cells],
                pressure_fluctuation_pa: vec![f64::MAX; cells],
                pressure_pa: vec![f64::MAX; cells],
                face_volume_flux_m3_per_s: vec![-f64::MAX; faces],
                outer_iterations: 123,
                linear_iterations: 1234567,
                residual: CanonicalFluidResidual {
                    boundary_balances: vec![
                            crate::physics::cfd::canonical::CanonicalFluidBoundaryBalance {
                                authority_id: "pressure_β".into(),
                                face_count: u64::MAX,
                                area_m2: f64::MAX,
                                volume_flow_rate_m3_per_s: -f64::MAX,
                                mass_flow_rate_kg_per_s: f64::MIN_POSITIVE,
                                advective_momentum_n: v,
                                viscous_traction_n: v,
                                pressure_traction_n: v,
                                pressure_fluctuation_traction_n: v,
                            };
                            faces
                        ],
                    face_forces: CanonicalFluidFaceForces {
                        advective_momentum_n: vec![v; faces],
                        viscous_traction_n: vec![v; faces],
                        pressure_traction_n: vec![v; faces],
                        pressure_fluctuation_traction_n: vec![v; faces],
                    },
                    cell_momentum_n: vec![v; cells],
                    cell_algebraic_momentum_n: vec![v; cells],
                    cell_momentum_relative: vec![f64::MAX; cells],
                    cell_continuity_m3_per_s: vec![-f64::MAX; cells],
                    cell_continuity_relative: vec![f64::MIN_POSITIVE; cells],
                    global_momentum_n: v,
                    global_momentum_relative: f64::MAX,
                    global_boundary_flux_m3_per_s: -f64::MAX,
                    momentum_relative: f64::MIN_POSITIVE,
                    continuity_relative: f64::MAX,
                    flux_constitutive_relative: -f64::MAX,
                },
            };
            assert_eq!(
                profile_serialized(&original, LIMITS).unwrap(),
                profile_serialized(
                    &original_profile(cells, faces, std::iter::repeat_n("pressure_β", faces)),
                    LIMITS
                )
                .unwrap(),
            );
        }
    }
}
