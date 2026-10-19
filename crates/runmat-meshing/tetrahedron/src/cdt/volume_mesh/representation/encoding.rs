use super::*;
use serde::{ser::SerializeSeq, Serializer};

pub(super) struct Entities<'a, 'b>(
    pub(super) &'a VolumeRepresentation<'b>,
    pub(super) &'a [PersistentEntityId],
);
impl Serialize for Entities<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut output = serializer.serialize_seq(Some(self.1.len()))?;
        for entity in self.1 {
            output.serialize_element(&self.0.entity::<S::Error>(entity)?)?;
        }
        output.end()
    }
}
struct Cells<'a, 'b>(&'a VolumeRepresentation<'b>);
impl Serialize for Cells<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut output = serializer.serialize_seq(Some(self.0.topology.tetrahedra.len()))?;
        for cell in &self.0.topology.tetrahedra {
            let region = cell
                .region_id
                .as_ref()
                .map(|region| self.0.entity::<S::Error>(region))
                .transpose()?;
            output.serialize_element(&(cell.vertex_indices, region))?;
        }
        output.end()
    }
}
struct Nodes<'a, 'b>(&'a VolumeRepresentation<'b>);
impl Serialize for Nodes<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut output = serializer.serialize_seq(Some(self.0.provenance.nodes.len()))?;
        for node in &self.0.provenance.nodes {
            output.serialize_element(&(
                self.0.node::<S::Error>(node.node_identity)?,
                Entities(self.0, &node.entity_ids),
            ))?;
        }
        output.end()
    }
}
struct Segments<'a, 'b>(&'a VolumeRepresentation<'b>);
impl Serialize for Segments<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut output = serializer.serialize_seq(Some(self.0.provenance.segments.len()))?;
        for edge in &self.0.provenance.segments {
            let nodes = edge.node_identities.map(|id| self.0.node::<S::Error>(id));
            let [first, second] = nodes;
            output.serialize_element(&(
                [first?, second?],
                Entities(self.0, &edge.entity_ids),
                edge.edge_parameters,
            ))?;
        }
        output.end()
    }
}
struct Facets<'a, 'b>(&'a VolumeRepresentation<'b>);
impl Serialize for Facets<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut output = serializer.serialize_seq(Some(self.0.provenance.facets.len()))?;
        for face in &self.0.provenance.facets {
            let [first, second, third] = face.node_identities.map(|id| self.0.node::<S::Error>(id));
            output.serialize_element(&(
                [first?, second?, third?],
                self.0.chart::<S::Error>(face.chart_id)?,
                Entities(self.0, &face.entity_ids),
                Entities(self.0, &face.region_ids),
            ))?;
        }
        output.end()
    }
}

impl Serialize for VolumeRepresentation<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Wire<'a, 'b> {
            nodes: &'b [DelaunayVolumeNode],
            entities: &'a [&'b PersistentEntityId],
            charts: &'a [StableDigest],
            cells: Cells<'a, 'b>,
            node_provenance: Nodes<'a, 'b>,
            segment_provenance: Segments<'a, 'b>,
            facet_provenance: Facets<'a, 'b>,
        }
        Wire {
            nodes: &self.topology.nodes,
            entities: &self.entities,
            charts: &self.charts,
            cells: Cells(self),
            node_provenance: Nodes(self),
            segment_provenance: Segments(self),
            facet_provenance: Facets(self),
        }
        .serialize(serializer)
    }
}
