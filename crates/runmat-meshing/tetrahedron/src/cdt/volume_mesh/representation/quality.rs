//! Lossless numeric quality rows. Node/region identity comes from the same
//! canonical cell; context, metric and source dictionaries preserve all evidence.
use super::*;
use crate::cdt::DelaunayTetrahedronQuality;
use runmat_meshing_size::metric::{MetricSourceKind, MetricTensor3};
use serde::{
    ser::{Error, SerializeSeq},
    Serializer,
};

// Context/metric/sources, contribution counts, geometric measures, boundary
// counts and violation measures. No float is quantized or regenerated here.
pub(in crate::cdt::volume_mesh) type QualityValues =
    (u32, u32, u32, [u32; 3], [f64; 5], [u8; 2], [f64; 2]);
fn metric_bits(metric: &MetricTensor3) -> [u64; 6] {
    [
        metric.xx, metric.yy, metric.zz, metric.xy, metric.xz, metric.yz,
    ]
    .map(f64::to_bits)
}
pub(in crate::cdt::volume_mesh) struct QualityTables<'a> {
    quality: &'a DelaunayVolumeQuality,
    metrics: Vec<&'a MetricTensor3>,
    contexts: Vec<&'a [PersistentEntityId]>,
    active_sources: Vec<&'a [MetricSourceKind]>,
}
impl<'a> QualityTables<'a> {
    pub fn preparation_bytes(
        quality: &DelaunayVolumeQuality,
    ) -> Result<u64, DelaunayVolumeMeshError> {
        (quality.tetrahedra.len() as u64)
            .checked_mul(
                (std::mem::size_of::<&MetricTensor3>()
                    + 2 * std::mem::size_of::<&[MetricSourceKind]>()
                    + std::mem::size_of::<&[PersistentEntityId]>()) as u64,
            )
            .ok_or_else(super::tables::overflow)
    }
    pub fn new(
        quality: &'a DelaunayVolumeQuality,
        control: &dyn runmat_canonical_codec::CanonicalCodecControl,
    ) -> Result<Self, DelaunayVolumeMeshError> {
        let count = quality.tetrahedra.len();
        let mut metrics = Vec::with_capacity(count);
        let mut contexts = Vec::with_capacity(count);
        let mut active_sources = Vec::with_capacity(count);
        for (index, cell) in quality.tetrahedra.iter().enumerate() {
            if index.is_multiple_of(128) {
                super::tables::checkpoint(control)?;
            }
            metrics.push(&cell.resolved_metric);
            contexts.push(cell.incident_metric_entity_ids.as_slice());
            active_sources.push(cell.active_metric_sources.as_slice());
        }
        super::tables::unique(
            &mut metrics,
            |left, right| metric_bits(left).cmp(&metric_bits(right)),
            control,
        )?;
        super::tables::unique(&mut contexts, Ord::cmp, control)?;
        super::tables::unique(&mut active_sources, Ord::cmp, control)?;
        Ok(Self {
            quality,
            metrics,
            contexts,
            active_sources,
        })
    }
}
pub(in crate::cdt::volume_mesh) struct QualityRepresentation<'a, 'b> {
    pub volume: &'a VolumeRepresentation<'b>,
    pub tables: &'a QualityTables<'b>,
}
struct Contexts<'a, 'b>(&'a QualityRepresentation<'a, 'b>);
impl Serialize for Contexts<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut output = serializer.serialize_seq(Some(self.0.tables.contexts.len()))?;
        for context in &self.0.tables.contexts {
            output.serialize_element(&super::encoding::Entities(self.0.volume, context))?;
        }
        output.end()
    }
}
struct Values<'a, 'b>(&'a QualityRepresentation<'a, 'b>);
impl Serialize for Values<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let tables = self.0.tables;
        let mut output = serializer.serialize_seq(Some(tables.quality.tetrahedra.len()))?;
        for cell in &tables.quality.tetrahedra {
            let context = tables
                .contexts
                .binary_search(&cell.incident_metric_entity_ids.as_slice())
                .map_err(|_| S::Error::custom("missing quality context"))?;
            let metric = tables
                .metrics
                .binary_search_by_key(&metric_bits(&cell.resolved_metric), |metric| {
                    metric_bits(metric)
                })
                .map_err(|_| S::Error::custom("missing quality metric"))?;
            let active = tables
                .active_sources
                .binary_search(&cell.active_metric_sources.as_slice())
                .map_err(|_| S::Error::custom("missing quality sources"))?;
            let row: QualityValues = (
                u32::try_from(context).map_err(S::Error::custom)?,
                u32::try_from(metric).map_err(S::Error::custom)?,
                u32::try_from(active).map_err(S::Error::custom)?,
                [
                    cell.applied_metric_contribution_count,
                    cell.clipped_metric_contribution_count,
                    cell.rejected_metric_contribution_count,
                ],
                [
                    cell.minimum_metric_edge_length,
                    cell.maximum_metric_edge_length,
                    cell.metric_circumradius,
                    cell.metric_radius_edge_ratio,
                    cell.metric_scaled_jacobian,
                ],
                [cell.boundary_node_count, cell.protected_facet_count],
                [
                    cell.refinement_violation_ratio,
                    cell.optimization_violation_ratio,
                ],
            );
            output.serialize_element(&row)?;
        }
        output.end()
    }
}
impl Serialize for QualityRepresentation<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let quality = self.tables.quality;
        let locate = |identity: Option<[StableDigest; 4]>| -> Result<Option<u32>, S::Error> {
            identity
                .map(|identity| {
                    let (index, _) = crate::cdt::topology::queries::find_oriented_tetrahedron(
                        self.volume.topology,
                        identity,
                    )
                    .ok_or_else(|| S::Error::custom("quality summary names a missing cell"))?;
                    u32::try_from(index).map_err(S::Error::custom)
                })
                .transpose()
        };
        #[derive(Serialize)]
        struct Wire<'a, 'b> {
            metrics: &'a [&'b MetricTensor3],
            contexts: Contexts<'a, 'b>,
            active_sources: &'a [&'b [MetricSourceKind]],
            values: Values<'a, 'b>,
            worsts: [Option<u32>; 2],
            bounds: [f64; 3],
        }
        Wire {
            metrics: &self.tables.metrics,
            contexts: Contexts(self),
            active_sources: &self.tables.active_sources,
            values: Values(self),
            worsts: [
                locate(quality.worst_refinement_tetrahedron)?,
                locate(quality.worst_optimization_tetrahedron)?,
            ],
            bounds: [
                quality.maximum_metric_edge_length,
                quality.maximum_radius_edge_ratio,
                quality.minimum_metric_scaled_jacobian,
            ],
        }
        .serialize(serializer)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::cdt::volume_mesh) struct DecodedQuality {
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    metrics: Vec<MetricTensor3>,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    contexts: Vec<runmat_canonical_codec::ExactVec<u32>>,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    active_sources: Vec<runmat_canonical_codec::ExactVec<MetricSourceKind>>,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    values: Vec<QualityValues>,
    worsts: [Option<u32>; 2],
    bounds: [f64; 3],
}
impl DecodedQuality {
    pub fn into_quality(
        self,
        parts: &mut DecodedVolumeParts<'_>,
        options: DelaunayVolumeMeshOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<DelaunayVolumeQuality, DelaunayVolumeMeshError> {
        let Self {
            metrics,
            contexts,
            active_sources,
            values,
            worsts,
            bounds,
        } = self;
        if values.len() != parts.topology.tetrahedra.len()
            || metrics
                .windows(2)
                .any(|pair| metric_bits(&pair[0]) >= metric_bits(&pair[1]))
            || contexts.windows(2).any(|pair| pair[0] >= pair[1])
            || active_sources.windows(2).any(|pair| pair[0] >= pair[1])
            || contexts
                .iter()
                .flatten()
                .any(|index| *index as usize >= parts.entities.len())
        {
            return Err(invalid(
                "indexed quality dictionaries or cell inventory differ",
            ));
        }
        parts
            .expansion
            .collection::<DelaunayTetrahedronQuality>(values.len())?;
        let mut tetrahedra = Vec::with_capacity(values.len());
        for (index, (context, metric, active, counts, geometry, boundary, violations)) in
            values.into_iter().enumerate()
        {
            if (index as u64).is_multiple_of(options.quality.cancellation_check_interval.max(1))
                && cancellation.is_cancelled()
            {
                return Err(error(
                    DelaunayVolumeMeshStage::Quality,
                    DelaunayVolumeMeshErrorKind::Cancelled,
                    "indexed quality decoding cancelled",
                ));
            }
            let cell = &parts.topology.tetrahedra[index];
            let region = cell
                .region_id
                .as_ref()
                .ok_or_else(|| invalid("quality cell has no region"))?;
            let context = contexts
                .get(context as usize)
                .ok_or_else(|| invalid("unknown quality context index"))?;
            let sources = active_sources
                .get(active as usize)
                .ok_or_else(|| invalid("unknown quality source index"))?;
            parts
                .expansion
                .collection::<MetricSourceKind>(sources.len())?;
            tetrahedra.push(DelaunayTetrahedronQuality {
                node_identities: cell
                    .vertex_indices
                    .map(|vertex| parts.topology.nodes[vertex as usize].identity),
                region_id: region.clone(),
                incident_metric_entity_ids: parts.expansion.entities(
                    context,
                    &parts.entities,
                    cancellation,
                )?,
                resolved_metric: *metrics
                    .get(metric as usize)
                    .ok_or_else(|| invalid("unknown quality metric index"))?,
                active_metric_sources: sources.to_vec(),
                applied_metric_contribution_count: counts[0],
                clipped_metric_contribution_count: counts[1],
                rejected_metric_contribution_count: counts[2],
                minimum_metric_edge_length: geometry[0],
                maximum_metric_edge_length: geometry[1],
                metric_circumradius: geometry[2],
                metric_radius_edge_ratio: geometry[3],
                metric_scaled_jacobian: geometry[4],
                boundary_node_count: boundary[0],
                protected_facet_count: boundary[1],
                refinement_violation_ratio: violations[0],
                optimization_violation_ratio: violations[1],
            });
        }
        let locate = |index: Option<u32>| {
            index
                .map(|index| {
                    tetrahedra
                        .get(index as usize)
                        .map(|cell| cell.node_identities)
                        .ok_or_else(|| invalid("unknown quality summary index"))
                })
                .transpose()
        };
        Ok(DelaunayVolumeQuality {
            worst_refinement_tetrahedron: locate(worsts[0])?,
            worst_optimization_tetrahedron: locate(worsts[1])?,
            tetrahedra,
            maximum_metric_edge_length: bounds[0],
            maximum_radius_edge_ratio: bounds[1],
            minimum_metric_scaled_jacobian: bounds[2],
        })
    }
}
