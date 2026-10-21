//! Native copies and canonical materialization admitted before publication allocation.
//! Numerical storage borrows original U/p/q and uses the core field wire authority.
use super::*;
use crate::{
    execution::SolverResultIdentity,
    results::field::{
        FiniteVolumeFieldTopology, NativeOrientedFace, RecoveredFieldArtifactEntry,
        RecoveredFieldEntityCounts,
    },
};
use runmat_canonical_codec::{profile_serialized, SerializedCanonicalProfile};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{ser::SerializeSeq, Serialize, Serializer};
use std::mem::size_of;

fn add(a: u64, b: u64) -> AnalysisExecutionResult<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("fluid native field residency overflow"))
}
fn vector_bytes<T>(count: usize) -> AnalysisExecutionResult<u64> {
    count
        .checked_mul(size_of::<T>())
        .and_then(|n| n.checked_add(size_of::<Vec<T>>()))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| invalid("fluid native typed allocation overflow"))
}
fn codec_bytes(profile: SerializedCanonicalProfile) -> AnalysisExecutionResult<u64> {
    add(
        profile.encoded_bytes_upper_bound,
        profile.materialization_resident_bytes_estimate,
    )
}

enum Values<'a> {
    Velocity(&'a [[f64; 3]]),
    Scalar(&'a [f64]),
}
impl Serialize for Values<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Velocity(values) => serializer.collect_seq(values.iter().flatten()),
            Self::Scalar(values) => values.serialize(serializer),
        }
    }
}
#[derive(Serialize)]
struct FieldWire<'a, S: Serialize> {
    identity: Digest,
    definition: &'a RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    storage: S,
}
struct Fields<'a> {
    definitions: &'a [RecoveredFieldDefinition],
    counts: &'a [usize],
    original: &'a crate::physics::cfd::canonical::CanonicalFluidResult,
    extra: Option<Values<'a>>,
}
impl Serialize for Fields<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut values = vec![
            Values::Velocity(&self.original.velocity_m_per_s),
            Values::Scalar(&self.original.pressure_pa),
            Values::Scalar(&self.original.face_volume_flux_m3_per_s),
        ];
        if let Some(extra) = &self.extra {
            values.push(match extra {
                Values::Velocity(v) => Values::Velocity(v),
                Values::Scalar(v) => Values::Scalar(v),
            });
        }
        let mut seq = serializer.serialize_seq(Some(self.definitions.len()))?;
        for ((definition, count), values) in self.definitions.iter().zip(self.counts).zip(values) {
            let shape = definition
                .storage_shape(*count)
                .map_err(serde::ser::Error::custom)?;
            let storage =
                AnalysisField::host_f64_serialization(definition.field_id(), &shape, &values);
            seq.serialize_element(&FieldWire {
                identity: Digest::from_bytes([0; 32]),
                definition,
                sample: RecoveredFieldSample::Final,
                storage,
            })?;
        }
        seq.end()
    }
}
#[derive(Serialize)]
struct ArtifactWire<'a> {
    schema_version: u16,
    identity: Digest,
    recovery_request: Digest,
    source: FieldRecoverySource,
    topology_digest: Digest,
    entity_counts: RecoveredFieldEntityCounts,
    fields: Fields<'a>,
}
fn artifact_profile(
    definitions: &[RecoveredFieldDefinition],
    counts: &[usize],
    original: &crate::physics::cfd::canonical::CanonicalFluidResult,
    extra: Option<Values<'_>>,
    entity_counts: RecoveredFieldEntityCounts,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    let expected = 3 + usize::from(extra.is_some());
    if definitions.len() != expected
        || counts.len() != expected
        || extra.as_ref().is_some_and(|values| match values {
            Values::Velocity(v) => v.len() != counts[3],
            Values::Scalar(v) => v.len() != counts[3],
        })
        || original.velocity_m_per_s.len() != counts[0]
        || original.pressure_pa.len() != counts[1]
        || original.pressure_fluctuation_pa.len() != counts[1]
        || original.face_volume_flux_m3_per_s.len() != counts[2]
    {
        return Err(invalid("fluid native profile original cardinality"));
    }
    profile_serialized(
        &ArtifactWire {
            schema_version: 1,
            identity: Digest::from_bytes([0; 32]),
            recovery_request: Digest::from_bytes([0; 32]),
            source: FieldRecoverySource::SolverResult(SolverResultIdentity::derive(
                "native-fluid-profile-placeholder",
                &Digest::from_bytes([0; 32]),
            )?),
            topology_digest: Digest::from_bytes([0; 32]),
            entity_counts,
            fields: Fields {
                definitions,
                counts,
                original,
                extra,
            },
        },
        LIMITS,
    )
    .map_err(invalid)
}

pub(super) fn requested_bytes(
    result: &FluidSteadySolverResult,
    workload: &FluidSteadyWorkload,
    source: &super::super::workload::PreparedFluidSource,
) -> AnalysisExecutionResult<u64> {
    let definitions = super::field_definition_inventory(workload.coupling())?;
    let nodes = source.source.nodes.len();
    let cells = source.topology.cells.len();
    let faces = source.topology.faces.len();
    let extra = match (result.conjugate_heat(), result.coupled_displacement()) {
        (Some(energy), _) => Some(Values::Scalar(&energy.temperature_k)),
        (_, Some(displacement)) => Some(Values::Velocity(displacement)),
        _ => None,
    };
    let extra_components = match &extra {
        Some(Values::Velocity(_)) => 3,
        Some(Values::Scalar(_)) => 1,
        None => 0,
    };
    let mut counts = vec![cells, cells, faces];
    if extra.is_some() {
        counts.push(nodes);
    }
    let entity_counts = RecoveredFieldEntityCounts {
        nodes,
        volume_elements: cells,
        oriented_faces: faces,
        volume_integration_points: 0,
        oriented_edges: 0,
        beam_elements: 0,
        shell_elements: 0,
        connectors: 0,
        constraint_groups: 0,
    };
    let artifact = artifact_profile(
        &definitions,
        &counts,
        result.original(),
        extra,
        entity_counts,
    )?;
    let catalog = RecoveredFieldArtifactEntry::profile_sampled_catalog_shape(
        &definitions,
        &counts,
        RecoveredFieldSample::Final,
        1,
    )
    .map_err(invalid)?;
    let mut required = add(
        source.preparation_resident_bytes,
        result.reservation().resident_bytes_upper_bound,
    )?;
    // The original FV validation rederives complete geometry while its source is
    // retained. Reserve that real shared geometry envelope again, not just layout vectors.
    required = add(
        required,
        u64::try_from(source.topology.reservation.scratch_bytes).map_err(invalid)?,
    )?;
    // The field-layout owner profiles its counted validation and one face identity codec;
    // no numerical projection or unrelated tree-index estimator is substituted here.
    required = add(
        required,
        FiniteVolumeFieldTopology::validation_workspace_bytes(nodes, cells, faces)
            .map_err(invalid)?,
    )?;
    for bytes in [
        vector_bytes::<StableDigest>(nodes)?,
        vector_bytes::<StableDigest>(cells)?,
        vector_bytes::<NativeOrientedFace>(faces)?,
        vector_bytes::<RecoveredField>(definitions.len())?,
        vector_bytes::<f64>(
            cells
                .checked_mul(3)
                .ok_or_else(|| invalid("fluid velocity count overflow"))?,
        )?,
        vector_bytes::<f64>(cells)?,
        vector_bytes::<f64>(faces)?,
        vector_bytes::<f64>(
            nodes
                .checked_mul(extra_components)
                .ok_or_else(|| invalid("coupled node field count overflow"))?,
        )?,
    ] {
        required = add(required, bytes)?;
    }
    // Full field schema includes owned identifier/shape metadata and original
    // numeric sequences. Materialization + canonical buffer coexist with copies.
    required = add(required, codec_bytes(artifact)?)?;
    required = add(required, codec_bytes(catalog)?)?;
    // Original retained result also coexists with the field copies and identity encoding.
    required = add(
        required,
        codec_bytes(profile_serialized(result, LIMITS).map_err(invalid)?)?,
    )?;
    if required > workload.resident_ceiling()? || required > isize::MAX as u64 {
        return Err(invalid(format!(
            "fluid native field requested heap {required} exceeds authored/numerical ceiling {}",
            workload.resident_ceiling()?
        )));
    }
    Ok(required)
}

pub(super) fn admit(
    result: &FluidSteadySolverResult,
    workload: &FluidSteadyWorkload,
    source: &super::super::workload::PreparedFluidSource,
) -> AnalysisExecutionResult<()> {
    requested_bytes(result, workload, source).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_core_storage_has_exact_actual_serializer_profile() {
        let original = [[f64::MAX, -f64::MAX, f64::MIN_POSITIVE], [1.0, 2.0, 3.0]];
        let values = Values::Velocity(&original);
        let shape = [2, 3];
        let borrowed = AnalysisField::host_f64_serialization("cfd.velocity", &shape, &values);
        let actual = AnalysisField::host_f64(
            "cfd.velocity",
            shape.to_vec(),
            original.iter().flatten().copied().collect(),
        );
        assert_eq!(
            profile_serialized(&borrowed, LIMITS).unwrap(),
            profile_serialized(&actual, LIMITS).unwrap()
        );
        let definition = fluid_native_field_definition_inventory().unwrap().remove(0);
        let owned =
            RecoveredField::new(definition.clone(), RecoveredFieldSample::Final, actual).unwrap();
        let field = FieldWire {
            identity: Digest::from_bytes([0; 32]),
            definition: &definition,
            sample: RecoveredFieldSample::Final,
            storage: borrowed,
        };
        assert_eq!(
            profile_serialized(&field, LIMITS).unwrap(),
            profile_serialized(&owned, LIMITS).unwrap()
        );
    }

    #[test]
    fn whole_native_artifact_projection_matches_actual_serializer() {
        use crate::physics::cfd::canonical::{
            CanonicalFluidFaceForces, CanonicalFluidResidual, CanonicalFluidResult,
        };
        let original = CanonicalFluidResult {
            numerical_input_identity: Digest::from_bytes([1; 32]),
            authored_authority_identity: Digest::from_bytes([2; 32]),
            velocity_m_per_s: vec![[1.0, -2.0, 3.0]; 2],
            pressure_fluctuation_pa: vec![4.0; 2],
            pressure_reference_pa: vec![0.0; 2],
            pressure_pa: vec![4.0; 2],
            face_volume_flux_m3_per_s: vec![5.0; 7],
            outer_iterations: 1,
            linear_iterations: 0,
            residual: CanonicalFluidResidual {
                boundary_balances: vec![],
                face_forces: CanonicalFluidFaceForces {
                    advective_momentum_n: vec![],
                    viscous_traction_n: vec![],
                    pressure_traction_n: vec![],
                    pressure_fluctuation_traction_n: vec![],
                },
                cell_momentum_n: vec![],
                cell_algebraic_momentum_n: vec![],
                cell_momentum_relative: vec![],
                cell_continuity_m3_per_s: vec![],
                cell_continuity_relative: vec![],
                global_momentum_n: [0.0; 3],
                global_momentum_relative: 0.0,
                global_boundary_flux_m3_per_s: 0.0,
                momentum_relative: 0.0,
                continuity_relative: 0.0,
                flux_constitutive_relative: 0.0,
            },
        };
        let counts = [2, 2, 7];
        let entity_counts = RecoveredFieldEntityCounts {
            nodes: 5,
            volume_elements: 2,
            oriented_faces: 7,
            volume_integration_points: 0,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        };
        let definitions = fluid_native_field_definition_inventory().unwrap();
        let mut values = [
            original
                .velocity_m_per_s
                .iter()
                .flatten()
                .copied()
                .collect(),
            original.pressure_pa.clone(),
            original.face_volume_flux_m3_per_s.clone(),
        ]
        .into_iter();
        let fields = definitions
            .iter()
            .zip(counts)
            .map(|(definition, count)| {
                RecoveredField::new(
                    definition.clone(),
                    RecoveredFieldSample::Final,
                    AnalysisField::host_f64(
                        definition.field_id(),
                        definition.storage_shape(count).unwrap(),
                        values.next().unwrap(),
                    ),
                )
                .unwrap()
            })
            .collect();
        let digest = Digest::sha256(b"serialization-only-original-native-fixture");
        let artifact = RecoveredFieldArtifact::new(
            crate::execution::FieldRecoveryRequestIdentity::derive("profile-test-request", &digest)
                .unwrap(),
            FieldRecoverySource::SolverResult(
                SolverResultIdentity::derive("profile-test-result", &digest).unwrap(),
            ),
            digest,
            entity_counts,
            fields,
        )
        .unwrap();
        let projected =
            artifact_profile(&definitions, &counts, &original, None, entity_counts).unwrap();
        assert_eq!(projected, profile_serialized(&artifact, LIMITS).unwrap());
        let bytes = artifact.canonical_bytes().unwrap();
        assert!(bytes.len() as u64 <= projected.encoded_bytes_upper_bound);
        assert!(
            runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(
                &bytes
            )
            .unwrap()
                <= projected.materialization_resident_bytes_estimate
        );
    }

    #[test]
    fn typed_layout_and_codec_arithmetic_reject_overflow() {
        assert!(vector_bytes::<NativeOrientedFace>(usize::MAX).is_err());
        assert!(add(u64::MAX, 1).is_err());
        assert!(codec_bytes(SerializedCanonicalProfile {
            encoded_bytes_upper_bound: u64::MAX,
            materialization_resident_bytes_estimate: 1
        })
        .is_err());
    }
}
