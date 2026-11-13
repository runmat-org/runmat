//! Prepare immutable provenance and raw metric resolution once. Scoped trials
//! update their complete grading dependency closure in the quality journal.
use super::*;
use runmat_meshing_size::metric::{ResolvedMetricEvaluation, ResolvedMetricField};

pub(in crate::cdt) struct PreparedLocalQuality<'a> {
    request: &'a MetricFieldRequest,
    #[cfg(test)]
    cell_evaluations: std::cell::Cell<u64>,
    #[cfg(test)]
    lifecycle: std::cell::Cell<(u64, u64)>,
    provenance: &'a DelaunayVolumeProvenance,
    options: DelaunayVolumeQualityOptions,
    metric: Option<ResolvedMetricEvaluation>,
    field: ResolvedMetricField,
    raw_cache: std::cell::RefCell<BTreeMap<Vec<PersistentEntityId>, ResolvedMetricEvaluation>>,
    boundary_nodes: BTreeSet<StableDigest>,
    protected_facets: super::super::ProtectedFacets,
    protected_segments: BTreeSet<[StableDigest; 2]>,
    node_size: super::super::node_size::NodeSizeField,
}

impl<'a> PreparedLocalQuality<'a> {
    pub(in crate::cdt) fn options(&self) -> DelaunayVolumeQualityOptions {
        self.options
    }
    pub(in crate::cdt) fn provenance(&self) -> &DelaunayVolumeProvenance {
        self.provenance
    }
    pub(in crate::cdt) fn prepare(
        request: &'a MetricFieldRequest,
        provenance: &'a DelaunayVolumeProvenance,
        options: DelaunayVolumeQualityOptions,
    ) -> Result<Self, DelaunayVolumeQualityError> {
        let field = ResolvedMetricField::new(request).map_err(metric_error)?;
        let metric = field.uniform_evaluation().map_err(metric_error)?;
        let node_size = super::super::node_size::NodeSizeField::new(request, provenance)?;
        Ok(Self {
            node_size,
            request,
            #[cfg(test)]
            cell_evaluations: std::cell::Cell::new(0),
            #[cfg(test)]
            lifecycle: std::cell::Cell::new((0, 0)),
            provenance,
            options,
            metric,
            field,
            raw_cache: std::cell::RefCell::new(BTreeMap::new()),
            boundary_nodes: boundary_node_identities(provenance),
            protected_facets: protected_facet_identities(provenance),
            protected_segments: provenance
                .segments
                .iter()
                .map(|segment| segment.node_identities)
                .collect(),
        })
    }

    #[cfg(test)]
    pub(in crate::cdt) fn boundary_nodes(&self) -> &BTreeSet<StableDigest> {
        &self.boundary_nodes
    }

    pub(in crate::cdt) fn protected_facets(&self) -> &BTreeSet<[StableDigest; 3]> {
        &self.protected_facets.facets
    }
    pub(in crate::cdt) fn request(&self) -> &MetricFieldRequest {
        self.request
    }
    pub(in crate::cdt) fn protected_edges(&self) -> &BTreeSet<[StableDigest; 2]> {
        &self.protected_facets.edges
    }
    pub(in crate::cdt) fn protected_segments(&self) -> &BTreeSet<[StableDigest; 2]> {
        &self.protected_segments
    }

    // Test observations belong to this calculation, independent of process-wide
    // tracing callsite interest and concurrent diagnostic subscribers.
    #[cfg(test)]
    pub(in crate::cdt) fn observe_lifecycle(&self, preparation: bool) {
        let (preparations, handoffs) = self.lifecycle.get();
        self.lifecycle.set((
            preparations + u64::from(preparation),
            handoffs + u64::from(!preparation),
        ));
    }
    #[cfg(test)]
    pub(in crate::cdt) fn lifecycle_counts(&self) -> (u64, u64) {
        self.lifecycle.get()
    }

    #[cfg(test)]
    pub(in crate::cdt) fn cell_evaluations(&self) -> u64 {
        self.cell_evaluations.get()
    }

    pub(super) fn evaluate(
        &self,
        topology: &(impl VolumeTopologyRead + ?Sized),
        index: u32,
    ) -> Result<Entry, DelaunayVolumeQualityError> {
        #[cfg(test)]
        self.cell_evaluations
            .set(self.cell_evaluations.get().saturating_add(1));
        let cell = topology.cell(index);
        let region_id = cell.region_id.clone().ok_or_else(|| {
            error(
                DelaunayVolumeQualityErrorKind::InvalidTopology,
                Some(index as usize),
                "every working cell must have one assigned region",
            )
        })?;
        let nodes = cell.vertex_indices.map(|vertex| topology.node(vertex));
        let identities = nodes.map(|node| node.identity);
        let context = crate::cdt::volume_provenance::derive_metric_context_for_corners(
            identities,
            region_id.clone(),
            self.provenance,
        );
        let metric = self.resolve_raw(&context.incident_entity_ids)?;
        let raw_length = metric
            .metric
            .conservative_minimum_length_m()
            .map_err(metric_error)?;
        let quality = self.evaluate_metric(topology, index, context.incident_entity_ids, metric)?;
        let mut geometry =
            nodes.map(|node| node.coordinates_m.canonical_value().map(ordered_float_key));
        geometry.sort_unstable();
        let grading = self.metric.is_none().then_some(super::grading::Dependency {
            slot: index,
            neighbors: cell.neighbors,
            raw_length,
            cap: raw_length,
        });
        Ok(Entry {
            quality,
            geometry,
            grading,
        })
    }

    /// A new interior Steiner point has no boundary provenance. Resolve each
    /// prospective conical child's actual three-corner scope, not its parent's.
    /// Grading rescales a tensor uniformly and cannot change its shape Jacobian;
    /// the complete trial still rederives graded size and all acceptance facts.
    pub(in crate::cdt) fn insertion_metric(
        &self,
        face: [StableDigest; 3],
        region: &PersistentEntityId,
    ) -> Result<MetricTensor3, DelaunayVolumeQualityError> {
        self.metric_for_corners([face[0], face[1], face[2], StableDigest::ZERO], region)
    }

    pub(in crate::cdt) fn metric_for_corners(
        &self,
        corners: [StableDigest; 4],
        region: &PersistentEntityId,
    ) -> Result<MetricTensor3, DelaunayVolumeQualityError> {
        let context = crate::cdt::volume_provenance::derive_metric_context_for_corners(
            corners,
            region.clone(),
            self.provenance,
        );
        Ok(self.resolve_raw(&context.incident_entity_ids)?.metric)
    }

    pub(super) fn resolve_raw(
        &self,
        entities: &[PersistentEntityId],
    ) -> Result<ResolvedMetricEvaluation, DelaunayVolumeQualityError> {
        if let Some(metric) = &self.metric {
            return Ok(metric.clone());
        }
        if let Some(metric) = self.raw_cache.borrow().get(entities) {
            return Ok(metric.clone());
        }
        let metric = self
            .field
            .resolve(&entities.iter().cloned().collect())
            .map_err(metric_error)?;
        // Bounded optional acceleration; identities and metric controls are fixed
        // for this preparation. Cache eviction cannot change numerical semantics.
        if entities.len() <= 64 {
            let mut cache = self.raw_cache.borrow_mut();
            if cache.len() >= 128 {
                cache.clear();
            }
            cache.insert(entities.to_vec(), metric.clone());
        }
        Ok(metric)
    }

    pub(super) fn evaluate_metric(
        &self,
        topology: &(impl VolumeTopologyRead + ?Sized),
        index: u32,
        entities: Vec<PersistentEntityId>,
        metric: ResolvedMetricEvaluation,
    ) -> Result<DelaunayTetrahedronQuality, DelaunayVolumeQualityError> {
        let cell = topology.cell(index);
        let nodes = cell.vertex_indices.map(|vertex| topology.node(vertex));
        let identities = nodes.map(|node| node.identity);
        let region_id = cell.region_id.clone().ok_or_else(|| {
            error(
                DelaunayVolumeQualityErrorKind::InvalidTopology,
                Some(index as usize),
                "working cell lacks a region",
            )
        })?;
        let corner_metrics = self
            .node_size
            .corner_metrics(topology.nodes(), nodes, &region_id)?;
        evaluation::evaluate_tetrahedron_nodes(
            nodes,
            index as usize,
            evaluation::TetrahedronEvaluationContext {
                corner_metrics,
                region_id,
                incident_metric_entity_ids: entities,
                metric,
                boundary_node_count: identities
                    .iter()
                    .filter(|id| self.boundary_nodes.contains(id))
                    .count() as u8,
                protected_facet_count: protected_facet_count(identities, &self.protected_facets),
                protected_edges: super::super::protected_edge_mask(
                    identities,
                    &self.protected_facets,
                ),
            },
            self.options,
        )
    }

    pub(in crate::cdt) fn replace_quality(
        &self,
        owner: &mut WorkingVolumeQuality,
        removed: &[CellKey],
        created: &[u32],
        topology: &(impl VolumeTopologyRead + ?Sized),
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), DelaunayVolumeQualityError> {
        checkpoint(0, self.options, cancellation)?;
        let final_count = owner
            .cells
            .len()
            .checked_sub(removed.len())
            .and_then(|count| count.checked_add(created.len()));
        if topology.nodes().len() as u64 > self.options.maximum_nodes
            || final_count.is_none_or(|count| count as u64 > self.options.maximum_tetrahedra)
        {
            return Err(error(
                DelaunayVolumeQualityErrorKind::ResourceLimit,
                None,
                "working cavity exceeds the current quality inventory limit",
            ));
        }
        let removed_keys = removed.iter().copied().collect::<BTreeSet<_>>();
        if removed_keys.len() != removed.len()
            || removed.iter().any(|key| !owner.cells.contains_key(key))
        {
            return Err(error(
                DelaunayVolumeQualityErrorKind::InvalidQuality,
                None,
                "working cavity names duplicate or absent quality entries",
            ));
        }
        if self.metric.is_none() {
            return self.replace_scoped(owner, removed, created, topology, cancellation);
        }
        let mut entries = BTreeMap::new();
        for (position, index) in created.iter().copied().enumerate() {
            checkpoint(position as u64, self.options, cancellation)?;
            let entry = self.evaluate(topology, index)?;
            let key = cell_key(entry.quality.node_identities);
            if (owner.cells.contains_key(&key) && !removed_keys.contains(&key))
                || entries.insert(key, entry).is_some()
            {
                return Err(error(
                    DelaunayVolumeQualityErrorKind::InvalidQuality,
                    Some(index as usize),
                    "working cavity would duplicate a retained quality entry",
                ));
            }
        }
        for key in removed {
            owner.replace(*key, None);
        }
        for (key, entry) in entries {
            owner.replace(key, Some(entry));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
