//! Borrowed shared field wire projection: no recovered buffers or semantic-state construction.
use super::super::acoustic::AcousticPhasorConvention;
use super::*;
use crate::excitation::{ProjectionScalar, ProjectionVector3};
use crate::execution::acoustic::{native_acoustic_entity_counts, native_acoustic_field_definition};
use crate::execution::{AcousticHarmonicSolverResult, FieldRecoveryRequest, FieldRecoverySource};
use crate::results::field::{
    RecoveredFieldDefinition, RecoveredFieldEntityCounts, RecoveredFieldSample,
};

#[derive(Serialize)]
struct AcousticProjection<'a> {
    frequency: runmat_analysis_core::Frequency,
    phasor_convention: AcousticPhasorConvention,
    ordered_node_ids: &'a [u64],
    ordered_element_ids: &'a [u64],
    recovery_request: &'a FieldRecoveryRequest,
    fields: ArtifactProjection<'a>,
}

#[derive(Serialize)]
struct ArtifactProjection<'a> {
    schema_version: u16,
    identity: Digest,
    recovery_request: crate::execution::FieldRecoveryRequestIdentity,
    source: FieldRecoverySource,
    topology_digest: Digest,
    entity_counts: RecoveredFieldEntityCounts,
    fields: Vec<FieldProjection<'a>>,
}

struct FieldProjection<'a> {
    definition: RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    values: NativeValues<'a>,
    shape: Vec<usize>,
}

impl Serialize for FieldProjection<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("identity", &Digest::sha256([]))?;
        map.serialize_entry("definition", &self.definition)?;
        map.serialize_entry("sample", &self.sample)?;
        map.serialize_entry(
            "storage",
            &AnalysisField::host_f64_serialization(
                self.definition.field_id(),
                &self.definition.layout(),
                &self.shape,
                &self.values,
            ),
        )?;
        map.end()
    }
}

enum NativeValues<'a> {
    Pressure(&'a [ProjectionScalar]),
    Velocity(&'a [ProjectionVector3]),
    Intensity(&'a [[f64; 3]]),
    PortIntensity(&'a [f64]),
}
impl Serialize for NativeValues<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Pressure(values) => {
                serializer.collect_seq(values.iter().flat_map(|p| [p.real, p.imaginary]))
            }
            Self::Velocity(values) => serializer.collect_seq(
                values
                    .iter()
                    .flat_map(|v| (0..3).flat_map(move |axis| [v.real[axis], v.imaginary[axis]])),
            ),
            Self::Intensity(values) => serializer.collect_seq(values.iter().flatten()),
            Self::PortIntensity(values) => serializer.collect_seq(values.iter()),
        }
    }
}

pub(super) fn profile(
    request: &SemanticStatePublicationRequest,
    result: &AcousticHarmonicSolverResult,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    result.validate_publication_request(request)?;
    let recovery_request = result.native_field_request_from_result()?;
    let sample = RecoveredFieldSample::FrequencyHertz(result.frequency().canonical_value());
    let fields = vec![
        FieldProjection {
            definition: native_acoustic_field_definition(
                "acoustic.pressure",
                &recovery_request.plan.requirements[0],
            )?,
            sample,
            values: NativeValues::Pressure(result.pressure_pa()),
            shape: vec![result.ordered_node_ids().len(), 2],
        },
        FieldProjection {
            definition: native_acoustic_field_definition(
                "acoustic.particle_velocity",
                &recovery_request.plan.requirements[1],
            )?,
            sample,
            values: NativeValues::Velocity(result.velocity_m_per_s()),
            shape: vec![result.ordered_element_ids().len(), 3, 2],
        },
        FieldProjection {
            definition: native_acoustic_field_definition(
                crate::execution::acoustic::ACOUSTIC_INTENSITY_FIELD_ID,
                &recovery_request.plan.requirements[2],
            )?,
            sample,
            values: NativeValues::Intensity(result.intensity_w_per_m2()),
            shape: vec![result.ordered_element_ids().len(), 3],
        },
        FieldProjection {
            definition: native_acoustic_field_definition(
                crate::execution::acoustic::ACOUSTIC_PORT_INTENSITY_FIELD_ID,
                &recovery_request.plan.requirements[3],
            )?,
            sample,
            values: NativeValues::PortIntensity(result.port_intensity_w_per_m2()),
            shape: vec![result.ordered_element_ids().len()],
        },
    ];
    let payload = AcousticProjection {
        frequency: result.frequency(),
        phasor_convention: AcousticPhasorConvention::PositiveAngularFrequency,
        ordered_node_ids: result.ordered_node_ids(),
        ordered_element_ids: result.ordered_element_ids(),
        recovery_request: &recovery_request,
        fields: ArtifactProjection {
            schema_version: 1,
            identity: Digest::sha256([]),
            recovery_request: recovery_request.identity(),
            source: recovery_request.source,
            topology_digest: request.topology_digest,
            entity_counts: native_acoustic_entity_counts(result),
            fields,
        },
    };
    profile_state_payload(
        request,
        &Tagged {
            kind: "acoustic",
            key: "state",
            value: &payload,
        },
    )
}
