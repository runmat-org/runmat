//! Version 4 shares identities and losslessly records quality values by cell.
use super::super::representation::quality::{DecodedQuality, QualityRepresentation, QualityTables};
use super::super::representation::{DecodedVolumeRepresentation, VolumeRepresentation};
use super::*;

/// Prepared borrowed dictionaries are reused by profiling and encoding.
pub struct VolumeMeshEncoding<'a> {
    mesh: &'a DelaunayVolumeMesh,
    volume: VolumeRepresentation<'a>,
    quality: QualityTables<'a>,
}
impl<'a> VolumeMeshEncoding<'a> {
    pub(super) fn preparation_bytes(
        mesh: &DelaunayVolumeMesh,
    ) -> Result<u64, DelaunayVolumeMeshCodecError> {
        VolumeRepresentation::preparation_bytes(
            &mesh.topology,
            &mesh.provenance,
            Some(&mesh.quality),
        )
        .and_then(|bytes| {
            bytes
                .checked_add(QualityTables::preparation_bytes(&mesh.quality)?)
                .ok_or_else(super::super::representation::tables::overflow)
        })
        .map_err(invalid_mesh)
    }
    pub(super) fn new(
        mesh: &'a DelaunayVolumeMesh,
        control: &dyn runmat_canonical_codec::CanonicalCodecControl,
    ) -> Result<Self, DelaunayVolumeMeshCodecError> {
        Ok(Self {
            mesh,
            volume: VolumeRepresentation::new(
                &mesh.topology,
                &mesh.provenance,
                Some(&mesh.quality),
                control,
            )
            .map_err(invalid_mesh)?,
            quality: QualityTables::new(&mesh.quality, control).map_err(invalid_mesh)?,
        })
    }
    pub fn encoding_profile(
        &self,
        control: &dyn runmat_canonical_codec::CanonicalCodecControl,
    ) -> Result<runmat_canonical_codec::CanonicalEncodingProfile, DelaunayVolumeMeshCodecError>
    {
        let control = super::control::Control::from_codec(control);
        runmat_canonical_codec::profile_canonical_encoding_with_control(
            self,
            CONTRACT_LIMITS,
            &control,
        )
        .map_err(|failure| control.classify(invalid_encoding(failure)))
    }
    pub fn encode(
        &self,
        control: &dyn runmat_canonical_codec::CanonicalCodecControl,
    ) -> Result<Vec<u8>, DelaunayVolumeMeshCodecError> {
        let control = super::control::Control::from_codec(control);
        runmat_canonical_codec::encode_contract_with_control(
            CODEC_PREFIX,
            CONTRACT_DOMAIN,
            self,
            CONTRACT_LIMITS,
            &control,
        )
        .map_err(|failure| control.classify(invalid_encoding(failure)))
    }
}
impl Serialize for VolumeMeshEncoding<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Mesh<'a, 'b> {
            volume: &'a VolumeRepresentation<'b>,
            quality: QualityRepresentation<'a, 'b>,
            facet_recovery_insertions: &'b [crate::cdt::DelaunayFacetSteinerInsertion],
            quality_mutations: &'b [crate::cdt::DelaunayVolumeQualityMutation],
        }
        #[derive(Serialize)]
        struct Envelope<'a, 'b> {
            schema_version: u16,
            mesh: Mesh<'a, 'b>,
        }
        Envelope {
            schema_version: DELAUNAY_VOLUME_MESH_SCHEMA_VERSION,
            mesh: Mesh {
                volume: &self.volume,
                quality: QualityRepresentation {
                    volume: &self.volume,
                    tables: &self.quality,
                },
                facet_recovery_insertions: &self.mesh.facet_recovery_insertions,
                quality_mutations: &self.mesh.quality_mutations,
            },
        }
        .serialize(serializer)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DecodedEnvelope {
    pub schema_version: u16,
    pub mesh: DecodedMesh,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DecodedMesh {
    volume: DecodedVolumeRepresentation,
    quality: DecodedQuality,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    facet_recovery_insertions: Vec<crate::cdt::DelaunayFacetSteinerInsertion>,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    quality_mutations: Vec<crate::cdt::DelaunayVolumeQualityMutation>,
}
impl DecodedMesh {
    pub fn into_mesh(
        self,
        options: DelaunayVolumeMeshOptions,
        cancellation: &dyn MeshingCancellationSignal,
        admitted_bytes: u64,
        allocations: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
    ) -> Result<DelaunayVolumeMesh, super::super::DelaunayVolumeMeshError> {
        let Self {
            volume,
            quality,
            facet_recovery_insertions,
            quality_mutations,
        } = self;
        let mut parts = volume.into_parts(options, cancellation, admitted_bytes, allocations)?;
        let quality = quality.into_quality(&mut parts, options, cancellation)?;
        Ok(DelaunayVolumeMesh {
            topology: parts.topology,
            provenance: parts.provenance,
            quality,
            facet_recovery_insertions,
            quality_mutations,
        })
    }
}

#[cfg(test)]
mod tests;
