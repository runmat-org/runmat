//! Lossless indexed transport shared by volume publication and quality diagnostics.
//! Incidence is derived on decoding; geometry, regions and provenance are retained.
use super::*;
use runmat_canonical_codec::CanonicalCodecControl;
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;
pub(super) mod tables;

mod decoding;
mod encoding;
pub(super) mod quality;
pub(super) use decoding::{DecodedVolumeParts, DecodedVolumeRepresentation};
pub(super) use decoding::{FacetRow, NodeRow, SegmentRow};

pub(super) struct VolumeRepresentation<'a> {
    pub topology: &'a DelaunayVolumeTopology,
    pub provenance: &'a DelaunayVolumeProvenance,
    pub entities: Vec<&'a PersistentEntityId>,
    charts: Vec<StableDigest>,
}
impl<'a> VolumeRepresentation<'a> {
    fn reference_count(
        topology: &DelaunayVolumeTopology,
        provenance: &DelaunayVolumeProvenance,
        quality: Option<&DelaunayVolumeQuality>,
    ) -> Result<usize, DelaunayVolumeMeshError> {
        let mut count = topology.tetrahedra.len();
        for length in provenance
            .nodes
            .iter()
            .map(|value| value.entity_ids.len())
            .chain(
                provenance
                    .segments
                    .iter()
                    .map(|value| value.entity_ids.len()),
            )
            .chain(
                provenance
                    .facets
                    .iter()
                    .flat_map(|value| [value.entity_ids.len(), value.region_ids.len()]),
            )
            .chain(quality.into_iter().flat_map(|quality| {
                quality
                    .tetrahedra
                    .iter()
                    .map(|value| value.incident_metric_entity_ids.len())
            }))
        {
            count = count.checked_add(length).ok_or_else(tables::overflow)?;
        }
        Ok(count)
    }
    pub fn preparation_bytes(
        topology: &DelaunayVolumeTopology,
        provenance: &DelaunayVolumeProvenance,
        quality: Option<&DelaunayVolumeQuality>,
    ) -> Result<u64, DelaunayVolumeMeshError> {
        let entities = (Self::reference_count(topology, provenance, quality)? as u64)
            .checked_mul(std::mem::size_of::<&PersistentEntityId>() as u64)
            .ok_or_else(tables::overflow)?;
        let charts = (provenance.facets.len() as u64)
            .checked_mul(std::mem::size_of::<StableDigest>() as u64)
            .ok_or_else(tables::overflow)?;
        entities
            .checked_add(charts)
            .and_then(|bytes| bytes.checked_add(entities.max(charts)))
            .ok_or_else(tables::overflow)
    }
    pub fn new(
        topology: &'a DelaunayVolumeTopology,
        provenance: &'a DelaunayVolumeProvenance,
        quality: Option<&'a DelaunayVolumeQuality>,
        control: &dyn CanonicalCodecControl,
    ) -> Result<Self, DelaunayVolumeMeshError> {
        let mut entities =
            Vec::with_capacity(Self::reference_count(topology, provenance, quality)?);
        let mut append = |entity: &'a PersistentEntityId| -> Result<(), DelaunayVolumeMeshError> {
            if entities.len().is_multiple_of(128) {
                tables::checkpoint(control)?;
            }
            entities.push(entity);
            Ok(())
        };
        for cell in &topology.tetrahedra {
            if let Some(region) = &cell.region_id {
                append(region)?;
            }
        }
        for node in &provenance.nodes {
            for entity in &node.entity_ids {
                append(entity)?;
            }
        }
        for edge in &provenance.segments {
            for entity in &edge.entity_ids {
                append(entity)?;
            }
        }
        for face in &provenance.facets {
            for entity in face.entity_ids.iter().chain(&face.region_ids) {
                append(entity)?;
            }
        }
        if let Some(quality) = quality {
            for cell in &quality.tetrahedra {
                for entity in &cell.incident_metric_entity_ids {
                    append(entity)?;
                }
            }
        }
        tables::unique(&mut entities, |left, right| left.cmp(right), control)?;
        let mut charts = Vec::with_capacity(provenance.facets.len());
        for (index, face) in provenance.facets.iter().enumerate() {
            if index.is_multiple_of(128) {
                tables::checkpoint(control)?;
            }
            charts.push(face.chart_id);
        }
        tables::unique(&mut charts, Ord::cmp, control)?;
        Ok(Self {
            topology,
            provenance,
            entities,
            charts,
        })
    }
    fn entity<E: serde::ser::Error>(&self, value: &PersistentEntityId) -> Result<u32, E> {
        let index = self
            .entities
            .binary_search(&value)
            .map_err(|_| E::custom("volume entity is absent from transport dictionary"))?;
        u32::try_from(index).map_err(E::custom)
    }
    fn node<E: serde::ser::Error>(&self, value: StableDigest) -> Result<u32, E> {
        let index = self
            .topology
            .nodes
            .binary_search_by_key(&value, |node| node.identity)
            .map_err(|_| E::custom("volume provenance names a missing node"))?;
        u32::try_from(index).map_err(E::custom)
    }
    fn chart<E: serde::ser::Error>(&self, value: StableDigest) -> Result<u32, E> {
        let index = self
            .charts
            .binary_search(&value)
            .map_err(|_| E::custom("volume chart is absent from transport dictionary"))?;
        u32::try_from(index).map_err(E::custom)
    }
}

pub(super) fn invalid(reason: impl Into<String>) -> DelaunayVolumeMeshError {
    error(
        DelaunayVolumeMeshStage::Quality,
        DelaunayVolumeMeshErrorKind::InvalidTopology,
        reason,
    )
}
