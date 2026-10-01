//! Canonical sampled-value identity and allocation-free field encoding projection.
use runmat_analysis_core::{PhysicalQuantityKind, SignalSampleApproach};
use runmat_canonical_codec::{
    encode_contract, profile_serialized, CanonicalCodecError, CanonicalLimits,
};
use runmat_execution::Digest;
use serde::{
    ser::{SerializeSeq, SerializeStruct},
    Serialize, Serializer,
};

use super::{ComplexVector3, ProjectionSample, RealizedExcitation};

const PREFIX: &[u8] = b"runmat-fea-structural-sampled-value/v1\0";
const DOMAIN: &str = "fea-structural-sampled-value";
const LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 1024, 1024, 32);

pub(crate) fn sampled_value_digest(
    load: &RealizedExcitation,
) -> Result<Digest, CanonicalCodecError> {
    encode_contract(
        PREFIX,
        DOMAIN,
        &(load.quantity, &load.sample, &load.kind),
        LIMITS,
    )
    .map(Digest::sha256)
}

/// Projects the same field structure without a field allocation or numerical sampling.
/// The longest accepted parameter name bounds every supported projection-sample metadata shape.
pub(crate) fn nodal_sample_encoding_workspace(
    mesh_digest: Digest,
    node_count: usize,
) -> Result<u64, CanonicalCodecError> {
    let sample = ProjectionSample::StudyParameter {
        parameter_id: "p".repeat(LIMITS.maximum_string_bytes),
        canonical_value: 0.0,
        approach: SignalSampleApproach::LeftLimit,
    };
    profile_serialized(
        &(
            PhysicalQuantityKind::BodyForceDensity,
            sample,
            ProjectedKind {
                mesh_digest,
                node_count,
            },
        ),
        LIMITS,
    )
    .map(|profile| profile.materialization_resident_bytes_estimate)
}

struct ProjectedKind {
    mesh_digest: Digest,
    node_count: usize,
}
impl Serialize for ProjectedKind {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serializer.serialize_struct("RealizedExcitationKind", 2)?;
        value.serialize_field("kind", "body_force_field")?;
        value.serialize_field(
            "value",
            &ProjectedField {
                mesh_digest: self.mesh_digest,
                node_count: self.node_count,
            },
        )?;
        value.end()
    }
}
struct ProjectedField {
    mesh_digest: Digest,
    node_count: usize,
}
impl Serialize for ProjectedField {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serializer.serialize_struct("NodalVectorField", 2)?;
        value.serialize_field("mesh_digest", &self.mesh_digest)?;
        value.serialize_field("values", &ProjectedRows(self.node_count))?;
        value.end()
    }
}
struct ProjectedRows(usize);
impl Serialize for ProjectedRows {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut values = serializer.serialize_seq(Some(self.0))?;
        for _ in 0..self.0 {
            values.serialize_element(&ComplexVector3::real([0.0; 3]))?;
        }
        values.end()
    }
}

#[cfg(test)]
mod tests;
