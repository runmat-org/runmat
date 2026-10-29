//! Actual retained input+response wire shape, profiled without allocating arrays.
//! Codec numeric-wide bounds cover all finite physical values, not sample zeros.
use super::*;
use crate::physics::maxwell::{
    ComplexPhasor, MaxwellCurrentSample, MaxwellDriveExcitation, MaxwellDriveResponse,
    MaxwellElementVector, MaxwellFieldSample, MaxwellVolumeData,
};
use crate::results::field::VolumeFieldSamplingRule;
use runmat_meshing_core::StableDigest;
use serde::{Serialize, Serializer};
struct Repeated<'a, T> {
    value: &'a T,
    count: usize,
}
impl<T: Serialize> Serialize for Repeated<'_, T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(std::iter::repeat_n(self.value, self.count))
    }
}
#[derive(Serialize)]
struct Drive<'a> {
    drive_id: &'a str,
    excitation: MaxwellDriveExcitation,
    conductance_s: f64,
    unit_generator_field: Repeated<'a, MaxwellElementVector>,
    unit_closure_current: Repeated<'a, MaxwellElementVector>,
}
#[derive(Serialize)]
struct Input<'a> {
    #[serde(with = "runmat_types::physical::canonical_storage")]
    angular_frequency_rad_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::AngularFrequency, f64>,
    sampling: crate::results::field::SampledTopologyShape,
    volumes: Repeated<'a, MaxwellVolumeData>,
    source_current: Repeated<'a, MaxwellCurrentSample>,
    grounded_faces: Repeated<'a, StableDigest>,
    drives: Vec<Drive<'a>>,
}
#[derive(Serialize)]
struct Response<'a> {
    input_identity: Digest,
    system_plan_identity: Digest,
    mesh_digest: StableDigest,
    projection_digest: Digest,
    sampling_digest: Digest,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    angular_frequency_rad_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::AngularFrequency, f64>,
    edge_node_identities: Repeated<'a, [StableDigest; 2]>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    edge_circulations_v: runmat_types::physical::Quantity<
        runmat_types::physical::kind::ElectricPotential,
        Repeated<'a, ComplexPhasor>,
    >,
    samples: Repeated<'a, MaxwellFieldSample>,
    drives: Vec<MaxwellDriveResponse>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    electric_energy_j: runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    magnetic_energy_j: runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    source_power_va: runmat_types::physical::Quantity<
        runmat_types::physical::kind::ApparentPower,
        ComplexPhasor,
    >,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    ohmic_power_w: runmat_types::physical::Quantity<runmat_types::physical::kind::Power, f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    reactive_power_var:
        runmat_types::physical::Quantity<runmat_types::physical::kind::ReactivePower, f64>,
    relative_physical_residual: f64,
    relative_complex_power_error: f64,
    relative_charge_continuity_residual: f64,
}
#[derive(Serialize)]
struct ResultProjection<'a> {
    source_workload: crate::execution::AnalysisWorkloadIdentity,
    producer: crate::execution::StepIdentity,
    formulation: crate::admission::AnalysisRunIdentity,
    relative_residual_target: f64,
    element_order: runmat_meshing_core::ElementOrder,
    publication_declarations: Vec<&'a runmat_analysis_core::StateOutput>,
    input: Input<'a>,
    response: Response<'a>,
}
pub(super) fn profile(
    w: &MaxwellHarmonicWorkload,
    m: &SolverMeshArtifact,
) -> Result<runmat_canonical_codec::SerializedCanonicalProfile, ResourcePlanningError> {
    let elements = m.topology.volume_elements.len();
    let edges = elements
        .checked_mul(6)
        .ok_or(ResourcePlanningError::Overflow("Maxwell edge slots"))?;
    let samples = elements
        .checked_mul(15)
        .ok_or(ResourcePlanningError::Overflow("Maxwell sample slots"))?;
    let stable = StableDigest::from_bytes([0; 32]);
    let digest = Digest::from_bytes([0; 32]);
    let pair = [stable; 2];
    let zero = ComplexPhasor::ZERO;
    let volume = MaxwellVolumeData {
        element_identity: stable,
        permeability_h_per_m: runmat_types::physical::MagneticPermeability::new(0.0),
        permittivity_f_per_m: runmat_types::physical::ElectricPermittivity::new(0.0),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(0.0),
    };
    let current = MaxwellCurrentSample {
        sample_identity: stable,
        current_density_a_per_m2: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::ElectricCurrentDensity,
            _,
        >::new([zero; 3]),
    };
    let field = MaxwellFieldSample {
        sample_identity: stable,
        element_identity: stable,
        ordinal: 0,
        position_m: [0.0; 3],
        volume_weight_m3: 0.0,
        electric_field_v_per_m: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::ElectricField,
            _,
        >::new([zero; 3]),
        flux_density_t: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::MagneticFluxDensity,
            _,
        >::new([zero; 3]),
        magnetic_field_a_per_m: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::MagneticFieldIntensity,
            _,
        >::new([zero; 3]),
        electric_displacement_c_per_m2: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::ElectricDisplacement,
            _,
        >::new([zero; 3]),
        conduction_current_density_a_per_m2: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::ElectricCurrentDensity,
            _,
        >::new([zero; 3]),
    };
    // Every electrical terminal constraint or terminal current can head one drive.
    let labels: Vec<String> = w
        .model
        .constraints
        .iter()
        .filter(|c| matches!(c, runmat_analysis_core::AnalysisConstraint::Electrical(_)))
        .map(|c| c.constraint_id().to_string())
        .chain(
            w.model
                .excitations
                .iter()
                .filter_map(|action| match action {
                    runmat_analysis_core::ExcitationAction::ElectricCurrent(value) => {
                        Some(value.excitation_id.0.clone())
                    }
                    _ => None,
                }),
        )
        .collect();
    let element_vector = MaxwellElementVector {
        element_identity: stable,
        vector: [0.0; 3],
    };
    let excitation = MaxwellDriveExcitation::Current { amperes: zero };
    let value = ResultProjection {
        source_workload: w.identity().map_err(invalid)?,
        producer: w.producer_identity().map_err(invalid)?,
        formulation: w.formulation(m).map_err(invalid)?,
        relative_residual_target: w.relative_residual_target,
        element_order: m.resolved_request.element_order,
        publication_declarations: w.model.sequence.steps[0]
            .state_outputs
            .iter()
            .filter(|d| d.persistence != runmat_analysis_core::StatePersistence::CheckpointOnly)
            .collect(),
        input: Input {
            angular_frequency_rad_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::AngularFrequency,
                _,
            >::new(0.0),
            sampling: crate::results::field::SampledTopologyShape::new(
                m.topology.nodes.len(),
                elements,
                VolumeFieldSamplingRule::TetrahedronKeastFifteen,
            ),
            volumes: Repeated {
                value: &volume,
                count: elements,
            },
            source_current: Repeated {
                value: &current,
                count: samples,
            },
            grounded_faces: Repeated {
                value: &stable,
                count: m.topology.boundary_faces.len(),
            },
            drives: labels
                .iter()
                .map(|label| Drive {
                    drive_id: label,
                    excitation,
                    conductance_s: 0.0,
                    unit_generator_field: Repeated {
                        value: &element_vector,
                        count: elements,
                    },
                    unit_closure_current: Repeated {
                        value: &element_vector,
                        count: elements,
                    },
                })
                .collect(),
        },
        response: Response {
            input_identity: digest,
            system_plan_identity: digest,
            mesh_digest: stable,
            projection_digest: digest,
            sampling_digest: digest,
            angular_frequency_rad_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::AngularFrequency,
                _,
            >::new(0.0),
            edge_node_identities: Repeated {
                value: &pair,
                count: edges,
            },
            edge_circulations_v: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::ElectricPotential,
                _,
            >::new(Repeated {
                value: &zero,
                count: edges,
            }),
            samples: Repeated {
                value: &field,
                count: samples,
            },
            drives: labels
                .iter()
                .map(|label| MaxwellDriveResponse {
                    drive_id: label.clone(),
                    voltage_v: runmat_types::physical::Quantity::new(zero),
                    current_a: runmat_types::physical::Quantity::new(zero),
                    complex_power_va: runmat_types::physical::Quantity::new(zero),
                })
                .collect(),
            electric_energy_j: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Energy,
                _,
            >::new(0.0),
            magnetic_energy_j: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Energy,
                _,
            >::new(0.0),
            source_power_va: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::ApparentPower,
                _,
            >::new(zero),
            ohmic_power_w:
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Power, _>::new(0.0),
            reactive_power_var: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::ReactivePower,
                _,
            >::new(0.0),
            relative_physical_residual: 0.0,
            relative_complex_power_error: 0.0,
            relative_charge_continuity_residual: 0.0,
        },
    };
    runmat_canonical_codec::profile_serialized_shape(&value, LIMITS).map_err(invalid)
}

impl<T: runmat_types::physical::QuantityStorage> runmat_types::physical::QuantityStorage
    for Repeated<'_, T>
{
    fn all_finite(&self) -> bool {
        self.value.all_finite()
    }
}
