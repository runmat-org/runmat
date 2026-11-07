//! The transport version is distinct from the unchanged semantic result version.
use super::*;
use super::{decoding::DecodedFace, encoding::FaceEncoding};
use crate::{ExactFaceMesh, ExactFacePartitionOutcome, ExactFacePartitionResult};
use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits, ExactVec};
use runmat_meshing_core::{
    CanonicalEntityRange, MeshingCanonicalCodecControl, MeshingPartitionDescriptor,
    MeshingPartitionKind,
};
use runmat_meshing_curve::SharedCurveSegmentSplit;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
pub(crate) const PREFIX: &[u8] = b"runmat-meshing-indexed-face-partition/v2\0";
const DOMAIN: &str = "exact-face-partition-result/indexed-v2";
const DEFAULT_LIMITS: CanonicalLimits =
    CanonicalLimits::platform_content(1024 * 1024, 64);

pub struct PartitionEncoding<'a> {
    schema_version: u16,
    partition: Cow<'a, MeshingPartitionDescriptor>,
    outcome: OutcomeEncoding<'a>,
}
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum OutcomeEncoding<'a> {
    Converged {
        faces: Vec<FaceEncoding<'a>>,
    },
    RequiresCurveSplits {
        splits: &'a [SharedCurveSegmentSplit],
    },
}
impl<'a> PartitionEncoding<'a> {
    pub(crate) fn new(
        result: &'a ExactFacePartitionResult,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<Self, ExactSurfaceMeshError> {
        let outcome = match &result.outcome {
            ExactFacePartitionOutcome::Converged { faces } => OutcomeEncoding::Converged {
                faces: {
                    let mut prepared = Vec::with_capacity(faces.len());
                    for face in faces {
                        prepared.push(FaceEncoding::new(face, control)?);
                    }
                    prepared
                },
            },
            ExactFacePartitionOutcome::RequiresCurveSplits { splits } => {
                OutcomeEncoding::RequiresCurveSplits { splits }
            }
        };
        Ok(Self {
            schema_version: result.schema_version,
            partition: Cow::Borrowed(&result.partition),
            outcome,
        })
    }
    pub(crate) fn single(
        mesh: &'a ExactFaceMesh,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<Self, ExactSurfaceMeshError> {
        Ok(Self {
            schema_version: crate::EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
            partition: Cow::Owned(single_partition(mesh)),
            outcome: OutcomeEncoding::Converged {
                faces: vec![FaceEncoding::new(mesh, control)?],
            },
        })
    }
    pub(crate) fn single_preparation_storage_bytes(
        mesh: &ExactFaceMesh,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<u64, ExactSurfaceMeshError> {
        let identity = &mesh.source_face_id;
        let heap = identity.assembly_path.iter().try_fold(
            identity
                .assembly_path
                .len()
                .checked_mul(std::mem::size_of::<String>())
                .and_then(|bytes| bytes.checked_add(identity.source_topology_id.len()))
                .ok_or_else(|| invalid("single face identity inventory overflow"))?,
            |sum, path| {
                sum.checked_add(path.len())
                    .ok_or_else(|| invalid("single face path inventory overflow"))
            },
        )? as u64;
        FaceEncoding::preparation_bytes(mesh, signal)?
            .checked_add(std::mem::size_of::<FaceEncoding>() as u64)
            .and_then(|bytes| heap.checked_mul(2).and_then(|heap| bytes.checked_add(heap)))
            .ok_or_else(|| invalid("single face preparation inventory overflow"))
    }
    pub(crate) fn preparation_bytes(&self) -> u64 {
        let descriptor = match &self.partition {
            Cow::Borrowed(_) => 0,
            Cow::Owned(partition) => partition
                .owned_buffer_bytes()
                .expect("prepared descriptor inventory is bounded"),
        };
        descriptor
            + match &self.outcome {
                OutcomeEncoding::Converged { faces } => {
                    faces
                        .iter()
                        .map(FaceEncoding::retained_preparation_bytes)
                        .sum::<u64>()
                        + faces.capacity() as u64 * std::mem::size_of::<FaceEncoding>() as u64
                }
                OutcomeEncoding::RequiresCurveSplits { .. } => 0,
            }
    }
    pub(crate) fn profile(
        &self,
        limits: CanonicalLimits,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<runmat_canonical_codec::CanonicalEncodingProfile, ExactSurfaceMeshError> {
        let mut profile = self.buffer_profile(limits, signal)?;
        profile.resident_workspace_bytes_estimate = profile
            .resident_workspace_bytes_estimate
            .checked_add(self.preparation_bytes())
            .ok_or_else(|| invalid("surface encoder workspace overflow"))?;
        Ok(profile)
    }
    fn buffer_profile(
        &self,
        limits: CanonicalLimits,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<runmat_canonical_codec::CanonicalEncodingProfile, ExactSurfaceMeshError> {
        let control = MeshingCanonicalCodecControl::new(signal);
        runmat_canonical_codec::profile_canonical_encoding_with_control(self, limits, &control)
            .map_err(|error| codec_error(error, &control))
    }
    pub(crate) fn bounded_combined_profile(
        &self,
        limits: CanonicalLimits,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<Option<runmat_canonical_codec::CanonicalEncodingProfile>, ExactSurfaceMeshError>
    {
        match self.profile(limits, signal) {
            Ok(profile) => Ok(Some(profile)),
            Err(error) if error.kind == ExactSurfaceMeshErrorKind::ResourceLimit => Ok(None),
            Err(error) => Err(error),
        }
    }
    /// Admission precedes dictionary allocation. Input is immutable while the
    /// prepared owner reuses these dictionaries for profiling and encoding.
    pub fn preparation_storage_bytes(
        result: &ExactFacePartitionResult,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<u64, ExactSurfaceMeshError> {
        match &result.outcome {
            ExactFacePartitionOutcome::Converged { faces } => faces.iter().try_fold(
                (faces.len() as u64)
                    .checked_mul(std::mem::size_of::<FaceEncoding>() as u64)
                    .ok_or_else(|| invalid("surface owner inventory overflow"))?,
                |sum, face| {
                    checkpoint(signal)?;
                    sum.checked_add(FaceEncoding::preparation_bytes(face, signal)?)
                        .ok_or_else(|| invalid("surface dictionary inventory overflow"))
                },
            ),
            ExactFacePartitionOutcome::RequiresCurveSplits { .. } => Ok(0),
        }
    }
    pub fn prepare(
        result: &'a ExactFacePartitionResult,
        topology: &runmat_geometry_core::ExactBRepTopology,
        curves: &runmat_meshing_curve::SharedCurveMesh,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<Self, ExactSurfaceMeshError> {
        crate::validate_exact_face_partition_result_with_cancellation(
            result, topology, curves, signal,
        )?;
        Self::new(result, signal)
    }
    /// Validated wire inventory, including typed decode and original rich expansion.
    /// This is an allocation bound, not a semantic or mesh-quality verdict.
    pub fn decoding_storage_bytes(
        bytes: &[u8],
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<u64, ExactSurfaceMeshError> {
        let control = MeshingCanonicalCodecControl::new(signal);
        if !bytes.starts_with(PREFIX) {
            return runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate([bytes], &control)
                .map_err(|error| codec_error(error, &control));
        }
        let value = runmat_canonical_codec::validated_contract_value(
            PREFIX,
            DOMAIN,
            bytes,
            DEFAULT_LIMITS,
            &control,
        )
        .map_err(|error| codec_error(error, &control))?;
        super::inventory::requested_bytes(value, &control)
            .map_err(|error| codec_error(error, &control))
    }
    /// Additional encoder workspace; prepared dictionary storage is separate.
    pub fn encoding_profile(
        &self,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<runmat_canonical_codec::CanonicalEncodingProfile, ExactSurfaceMeshError> {
        self.buffer_profile(DEFAULT_LIMITS, signal)
    }
    pub fn encode_prepared(
        &self,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, ExactSurfaceMeshError> {
        self.encode(DEFAULT_LIMITS, signal)
    }
    /// Reuse the bound derived from this immutable prepared owner. Encoding still
    /// enforces the original codec limits; an inconsistent bound rejects.
    pub fn encode_profiled(
        &self,
        profile: &runmat_canonical_codec::CanonicalEncodingProfile,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, ExactSurfaceMeshError> {
        let control = MeshingCanonicalCodecControl::new(signal);
        let capacity = usize::try_from(profile.encoded_bytes_upper_bound)
            .map_err(|_| invalid("profiled surface output capacity overflow"))?;
        let bytes = runmat_canonical_codec::encode_contract_with_capacity(
            PREFIX,
            DOMAIN,
            self,
            DEFAULT_LIMITS,
            &control,
            capacity,
        )
        .map_err(|error| codec_error(error, &control))?;
        if bytes.len() > capacity || bytes.capacity() > capacity {
            return Err(invalid("surface output exceeds its prepared profile"));
        }
        Ok(bytes)
    }
    pub fn bounded_encoding_profile(
        &self,
        maximum_encoded_bytes: usize,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<Option<runmat_canonical_codec::CanonicalEncodingProfile>, ExactSurfaceMeshError>
    {
        let limits = CanonicalLimits {
            maximum_encoded_bytes: maximum_encoded_bytes.min(DEFAULT_LIMITS.maximum_encoded_bytes),
            ..DEFAULT_LIMITS
        };
        match self.buffer_profile(limits, signal) {
            Ok(profile) => Ok(Some(profile)),
            Err(error) if error.kind == ExactSurfaceMeshErrorKind::ResourceLimit => Ok(None),
            Err(error) => Err(error),
        }
    }
    pub fn encode_bounded(
        &self,
        maximum_encoded_bytes: usize,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, ExactSurfaceMeshError> {
        self.encode(
            CanonicalLimits {
                maximum_encoded_bytes: maximum_encoded_bytes
                    .min(DEFAULT_LIMITS.maximum_encoded_bytes),
                ..DEFAULT_LIMITS
            },
            signal,
        )
    }
    pub(crate) fn encode(
        &self,
        limits: CanonicalLimits,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, ExactSurfaceMeshError> {
        let control = MeshingCanonicalCodecControl::new(signal);
        runmat_canonical_codec::encode_contract_with_control(PREFIX, DOMAIN, self, limits, &control)
            .map_err(|error| codec_error(error, &control))
    }
}
impl Serialize for PartitionEncoding<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Envelope<'a, 'b> {
            format_version: u16,
            schema_version: u16,
            partition: &'a MeshingPartitionDescriptor,
            outcome: &'a OutcomeEncoding<'b>,
        }
        Envelope {
            format_version: 2,
            schema_version: self.schema_version,
            partition: &self.partition,
            outcome: &self.outcome,
        }
        .serialize(s)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecodedPartition {
    format_version: u16,
    schema_version: u16,
    partition: MeshingPartitionDescriptor,
    outcome: DecodedOutcome,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum DecodedOutcome {
    Converged {
        faces: ExactVec<DecodedFace>,
    },
    RequiresCurveSplits {
        splits: ExactVec<SharedCurveSegmentSplit>,
    },
}
pub(crate) fn decode(
    bytes: &[u8],
    limits: CanonicalLimits,
    signal: &dyn MeshingCancellationSignal,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    decode_admitted(bytes, limits, signal, |_| Ok(()))
}
pub(crate) fn decode_admitted(
    bytes: &[u8],
    limits: CanonicalLimits,
    signal: &dyn MeshingCancellationSignal,
    admit: impl FnOnce(u64) -> Result<(), CanonicalCodecError>,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    let control = MeshingCanonicalCodecControl::new(signal);
    let decoded: DecodedPartition = runmat_canonical_codec::decode_contract_with_admission(
        PREFIX,
        DOMAIN,
        bytes,
        limits,
        &control,
        |value| admit(super::inventory::requested_bytes(value, &control)?),
    )
    .map_err(|error| codec_error(error, &control))?;
    if decoded.format_version != 2 {
        return Err(invalid("unsupported indexed surface transport version"));
    }
    let outcome = match decoded.outcome {
        DecodedOutcome::Converged { faces } => ExactFacePartitionOutcome::Converged {
            faces: {
                let mut meshes = Vec::with_capacity(faces.len());
                for face in faces {
                    meshes.push(face.into_mesh(signal, |_| Ok(()))?);
                }
                meshes
            },
        },
        DecodedOutcome::RequiresCurveSplits { splits } => {
            ExactFacePartitionOutcome::RequiresCurveSplits {
                splits: splits.into_vec(),
            }
        }
    };
    Ok(ExactFacePartitionResult {
        schema_version: decoded.schema_version,
        partition: decoded.partition,
        outcome,
    })
}
pub(crate) fn single_partition(mesh: &ExactFaceMesh) -> MeshingPartitionDescriptor {
    MeshingPartitionDescriptor {
        kind: MeshingPartitionKind::CanonicalEntityBatch,
        partition_index: 0,
        partition_count: 1,
        entity_range: Some(CanonicalEntityRange {
            first: mesh.source_face_id.clone(),
            last: mesh.source_face_id.clone(),
            entity_count: 1,
        }),
    }
}
fn codec_error(
    error: CanonicalCodecError,
    control: &MeshingCanonicalCodecControl<'_>,
) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(
        if control.observed_cancellation() {
            ExactSurfaceMeshErrorKind::Cancelled
        } else if error.kind == runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit {
            ExactSurfaceMeshErrorKind::ResourceLimit
        } else {
            ExactSurfaceMeshErrorKind::InvalidEncoding
        },
        error.to_string(),
    )
}
