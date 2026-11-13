use super::*;
use crate::cdt::{
    DelaunayFacetProvenance, DelaunayNodeProvenance, DelaunaySegmentProvenance,
    DelaunayVolumeTetrahedron,
};

use runmat_canonical_codec::ExactVec;
pub(in crate::cdt::volume_mesh) type NodeRow = (u32, ExactVec<u32>);
pub(in crate::cdt::volume_mesh) type SegmentRow = ([u32; 2], ExactVec<u32>, [f64; 2]);
pub(in crate::cdt::volume_mesh) type FacetRow = ([u32; 3], u32, ExactVec<u32>, ExactVec<u32>);
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::cdt::volume_mesh) struct DecodedVolumeRepresentation {
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    nodes: Vec<DelaunayVolumeNode>,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    entities: Vec<PersistentEntityId>,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    charts: Vec<StableDigest>,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    cells: Vec<([u32; 4], Option<u32>)>,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    node_provenance: Vec<NodeRow>,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    segment_provenance: Vec<SegmentRow>,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    facet_provenance: Vec<FacetRow>,
}

/// Bound owned dictionary expansion within the already admitted codec window.
/// This is not a measurement of allocator or global validation scratch memory.
pub(in crate::cdt::volume_mesh) struct ExpansionBudget<'a> {
    remaining: u64,
    allocations: Option<&'a dyn runmat_geometry_core::GeometryEvaluationControl>,
}
impl ExpansionBudget<'_> {
    pub fn charge(&mut self, bytes: u64) -> Result<(), DelaunayVolumeMeshError> {
        let remaining = self.remaining.checked_sub(bytes).ok_or_else(|| {
            error(
                DelaunayVolumeMeshStage::Quality,
                DelaunayVolumeMeshErrorKind::ResourceLimit,
                "indexed volume expands beyond its admitted canonical materialization window",
            )
        })?;
        if let Some(control) = self.allocations {
            control.consume_allocation_bytes(bytes).map_err(|failure| {
                error(
                    DelaunayVolumeMeshStage::Quality,
                    DelaunayVolumeMeshErrorKind::ResourceLimit,
                    failure.to_string(),
                )
            })?;
        }
        self.remaining = remaining;
        Ok(())
    }
    pub fn collection<T>(&mut self, count: usize) -> Result<(), DelaunayVolumeMeshError> {
        self.charge(
            (count as u64)
                .checked_mul(std::mem::size_of::<T>() as u64)
                .ok_or_else(|| invalid("volume expansion inventory overflow"))?,
        )
    }
    pub fn entities(
        &mut self,
        indices: &[u32],
        table: &[PersistentEntityId],
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<PersistentEntityId>, DelaunayVolumeMeshError> {
        self.collection::<PersistentEntityId>(indices.len())?;
        collect_exact(indices.iter().enumerate().map(|(ordinal, index)| {
            if ordinal.is_multiple_of(128) && cancellation.is_cancelled() {
                return Err(error(
                    DelaunayVolumeMeshStage::Quality,
                    DelaunayVolumeMeshErrorKind::Cancelled,
                    "volume dictionary expansion cancelled",
                ));
            }
            let entity = table
                .get(*index as usize)
                .ok_or_else(|| invalid("unknown volume entity index"))?;
            // Handles share the dictionary allocation; the slot is charged above.
            Ok(entity.clone())
        }))
    }
}

pub(in crate::cdt::volume_mesh) struct DecodedVolumeParts<'a> {
    pub topology: DelaunayVolumeTopology,
    pub provenance: DelaunayVolumeProvenance,
    pub entities: Vec<PersistentEntityId>,
    pub expansion: ExpansionBudget<'a>,
}
impl DecodedVolumeRepresentation {
    pub fn into_parts<'a>(
        self,
        options: DelaunayVolumeMeshOptions,
        cancellation: &dyn MeshingCancellationSignal,
        admitted_bytes: u64,
        allocations: Option<&'a dyn runmat_geometry_core::GeometryEvaluationControl>,
    ) -> Result<DecodedVolumeParts<'a>, DelaunayVolumeMeshError> {
        let Self {
            nodes,
            entities,
            charts,
            cells,
            node_provenance,
            segment_provenance,
            facet_provenance,
        } = self;
        if entities.windows(2).any(|pair| pair[0] >= pair[1])
            || charts.windows(2).any(|pair| pair[0] >= pair[1])
            || charts.contains(&StableDigest::ZERO)
        {
            return Err(invalid(
                "indexed volume dictionaries must be canonical and unique",
            ));
        }
        if nodes.len() as u64 > options.quality.maximum_nodes
            || cells.len() as u64 > options.quality.maximum_tetrahedra
            || node_provenance.len() as u64 > options.provenance.maximum_node_bindings
            || segment_provenance.len() as u64 > options.provenance.maximum_segment_bindings
            || facet_provenance.len() as u64 > options.provenance.maximum_facet_bindings
        {
            return Err(error(
                DelaunayVolumeMeshStage::Quality,
                DelaunayVolumeMeshErrorKind::ResourceLimit,
                "indexed volume exceeds original inventory controls",
            ));
        }
        let poll = |index: usize| -> Result<(), DelaunayVolumeMeshError> {
            if (index as u64).is_multiple_of(options.quality.cancellation_check_interval.max(1))
                && cancellation.is_cancelled()
            {
                return Err(error(
                    DelaunayVolumeMeshStage::Quality,
                    DelaunayVolumeMeshErrorKind::Cancelled,
                    "indexed volume decoding cancelled",
                ));
            }
            Ok(())
        };
        let mut previous = None;
        for (index, (vertices, _)) in cells.iter().enumerate() {
            poll(index)?;
            let mut key = *vertices;
            key.sort_unstable();
            if key.iter().any(|index| *index as usize >= nodes.len())
                || previous.is_some_and(|old| old >= key)
            {
                return Err(invalid(
                    "indexed cells are not canonical or reference missing nodes",
                ));
            }
            previous = Some(key);
        }
        let mut expansion = ExpansionBudget {
            remaining: admitted_bytes,
            allocations,
        };
        expansion.collection::<DelaunayVolumeTetrahedron>(cells.len())?;
        let cells = collect_exact(cells.into_iter().enumerate().map(
            |(index, (vertices, region))| {
                poll(index)?;
                let region = region
                    .map(|index| {
                        let entity = entities
                            .get(index as usize)
                            .ok_or_else(|| invalid("unknown cell region index"))?;
                        Ok::<_, DelaunayVolumeMeshError>(entity.clone())
                    })
                    .transpose()?;
                Ok((vertices, region))
            },
        ))?;
        let topology = build_delaunay_volume_topology_with_regions(
            nodes,
            cells,
            options.refinement.step.insertion.topology,
            cancellation,
        )
        .map_err(|failure| {
            error(
                DelaunayVolumeMeshStage::Quality,
                match failure.kind {
                    crate::cdt::DelaunayTopologyErrorKind::ResourceLimit => {
                        DelaunayVolumeMeshErrorKind::ResourceLimit
                    }
                    crate::cdt::DelaunayTopologyErrorKind::Cancelled => {
                        DelaunayVolumeMeshErrorKind::Cancelled
                    }
                    _ => DelaunayVolumeMeshErrorKind::InvalidTopology,
                },
                failure.to_string(),
            )
        })?;
        let node = |index: u32| {
            topology
                .nodes
                .get(index as usize)
                .map(|node| node.identity)
                .ok_or_else(|| invalid("unknown provenance node index"))
        };
        expansion.collection::<DelaunayNodeProvenance>(node_provenance.len())?;
        expansion.collection::<DelaunaySegmentProvenance>(segment_provenance.len())?;
        expansion.collection::<DelaunayFacetProvenance>(facet_provenance.len())?;
        let nodes = collect_exact(node_provenance.into_iter().enumerate().map(
            |(index, (identity, refs))| {
                poll(index)?;
                Ok(DelaunayNodeProvenance {
                    node_identity: node(identity)?,
                    entity_ids: expansion.entities(&refs, &entities, cancellation)?,
                })
            },
        ))?;
        let segments = collect_exact(segment_provenance.into_iter().enumerate().map(
            |(index, (vertices, refs, edge_parameters))| {
                poll(index)?;
                Ok(DelaunaySegmentProvenance {
                    node_identities: [node(vertices[0])?, node(vertices[1])?],
                    entity_ids: expansion.entities(&refs, &entities, cancellation)?,
                    edge_parameters,
                })
            },
        ))?;
        let facets = collect_exact(facet_provenance.into_iter().enumerate().map(
            |(index, (vertices, chart, refs, regions))| {
                poll(index)?;
                Ok(DelaunayFacetProvenance {
                    node_identities: [node(vertices[0])?, node(vertices[1])?, node(vertices[2])?],
                    chart_id: *charts
                        .get(chart as usize)
                        .ok_or_else(|| invalid("unknown provenance chart index"))?,
                    entity_ids: expansion.entities(&refs, &entities, cancellation)?,
                    region_ids: expansion.entities(&regions, &entities, cancellation)?,
                })
            },
        ))?;
        Ok(DecodedVolumeParts {
            topology,
            provenance: DelaunayVolumeProvenance {
                nodes,
                segments,
                facets,
            },
            entities,
            expansion,
        })
    }
}

/// Fallible iterator collection must preserve the inventory admitted above.
/// Result's collecting adapter can lose the exact lower size hint and grow a
/// minimum-capacity Vec, particularly for the many one/two-entry dictionaries.
fn collect_exact<T>(
    items: impl ExactSizeIterator<Item = Result<T, DelaunayVolumeMeshError>>,
) -> Result<Vec<T>, DelaunayVolumeMeshError> {
    let mut result = Vec::new();
    result.try_reserve_exact(items.len()).map_err(|_| {
        error(
            DelaunayVolumeMeshStage::Quality,
            DelaunayVolumeMeshErrorKind::ResourceLimit,
            "cannot reserve the admitted volume expansion inventory",
        )
    })?;
    for item in items {
        result.push(item?);
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
