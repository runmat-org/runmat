//! Scoped grading state follows the quality journal. Only changed graph support
//! and its exact cap dependency closure participate in a speculative update.
use super::*;
use runmat_meshing_size::grading::{plan_metric_cap_update, MetricCapGraph, MetricCapUpdateError};

#[derive(Clone, PartialEq)]
pub(super) struct Dependency {
    pub slot: u32,
    pub neighbors: [Option<u32>; 4],
    pub raw_length: f64,
    pub cap: f64,
}
struct Graph<'a, T: VolumeTopologyRead + ?Sized> {
    owner: &'a WorkingVolumeQuality,
    created: &'a BTreeMap<u32, Entry>,
    removed: &'a BTreeMap<u32, CellKey>,
    topology: &'a T,
}
impl<T: VolumeTopologyRead + ?Sized> Graph<'_, T> {
    fn previous(&self, slot: u32) -> Option<&Entry> {
        // Changed slots carry their old identity in the bounded cavity overlay.
        // Every other slot still resolves through the shared current topology.
        // No second mesh-sized slot-to-identity inventory is retained.
        let key = self.removed.get(&slot).copied().or_else(|| {
            self.topology
                .contains_cell(slot)
                .then(|| self.topology.cell_key(slot))
        })?;
        self.owner.cells.get(&key)
    }
}
impl<T: VolumeTopologyRead + ?Sized> MetricCapGraph<u32> for Graph<'_, T> {
    fn previous_cap(&self, slot: u32) -> Option<f64> {
        self.previous(slot)
            .map(|entry| entry.grading.as_ref().unwrap().cap)
    }
    fn raw_length(&self, slot: u32) -> Result<Option<f64>, MetricContractError> {
        if let Some(entry) = self.created.get(&slot) {
            return Ok(Some(entry.grading.as_ref().unwrap().raw_length));
        }
        if self.removed.contains_key(&slot) {
            return Ok(None);
        }
        Ok(self
            .previous(slot)
            .map(|entry| entry.grading.as_ref().unwrap().raw_length))
    }
    fn neighbors(&self, slot: u32, previous: bool) -> Vec<u32> {
        if previous {
            self.previous(slot)
                .into_iter()
                .flat_map(|entry| {
                    entry
                        .grading
                        .as_ref()
                        .unwrap()
                        .neighbors
                        .into_iter()
                        .flatten()
                })
                .collect()
        } else {
            self.topology
                .tetrahedron(slot)
                .into_iter()
                .flat_map(|cell| cell.neighbors.into_iter().flatten())
                .collect()
        }
    }
}

impl PreparedLocalQuality<'_> {
    pub(in crate::cdt) fn initialize_scoped(
        &self,
        owner: &mut WorkingVolumeQuality,
        topology: &(impl VolumeTopologyRead + ?Sized),
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), DelaunayVolumeQualityError> {
        if self.request().contributions.is_empty() {
            return Ok(());
        }
        assert!(
            owner.journals.is_empty() && owner.baseline_originals.is_empty(),
            "scoped preparation requires an unedited owner"
        );
        owner.comparison_ready = false;
        let mut changed = Vec::with_capacity(owner.cells.len());
        for (position, slot) in (0..topology.tetrahedron_slot_count() as u32)
            .filter(|slot| topology.contains_cell(*slot))
            .enumerate()
        {
            checkpoint(position as u64, self.options(), cancellation)?;
            let key = topology.cell_key(slot);
            let entry = owner.cells.get_mut(&key).expect("admitted working cell");
            let raw = self.resolve_raw(&entry.quality.incident_metric_entity_ids)?;
            let length = raw
                .metric
                .conservative_minimum_length_m()
                .map_err(metric_error)?;
            entry.grading = Some(Dependency {
                slot,
                neighbors: topology.cell(slot).neighbors,
                raw_length: length,
                cap: length,
            });
            changed.push(slot);
        }
        let update = plan_metric_cap_update(
            self.request().maximum_grading_ratio,
            &Graph {
                owner,
                created: &BTreeMap::new(),
                removed: &BTreeMap::new(),
                topology,
            },
            changed,
            || cancellation.is_cancelled(),
        )
        .map_err(update_error)?;
        for (slot, cap) in update.caps {
            let key = topology.cell_key(slot);
            let entry = owner.cells.get_mut(&key).expect("admitted working cell");
            let mut raw = self.resolve_raw(&entry.quality.incident_metric_entity_ids)?;
            if cap < entry.grading.as_ref().unwrap().raw_length {
                runmat_meshing_size::grading::apply_metric_grading_cap(&mut raw, cap)
                    .map_err(metric_error)?;
            }
            if !metric_identity(&entry.quality, &raw) {
                return Err(error(
                    DelaunayVolumeQualityErrorKind::InvalidQuality,
                    Some(slot as usize),
                    "initial scoped quality differs from exact raw-metric graph grading",
                ));
            }
            entry.grading.as_mut().unwrap().cap = cap;
        }
        owner.reset_comparison();
        Ok(())
    }

    pub(super) fn replace_scoped(
        &self,
        owner: &mut WorkingVolumeQuality,
        removed: &[CellKey],
        created: &[u32],
        topology: &(impl VolumeTopologyRead + ?Sized),
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), DelaunayVolumeQualityError> {
        let removed_slots = removed
            .iter()
            .map(|key| {
                owner.cells[key]
                    .grading
                    .as_ref()
                    .expect("initialized scoped quality")
                    .slot
            })
            .zip(removed.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let mut entries = BTreeMap::new();
        let mut changed = removed_slots.keys().copied().collect::<BTreeSet<_>>();
        for &key in removed_slots.values() {
            changed.extend(
                owner.cells[&key]
                    .grading
                    .as_ref()
                    .unwrap()
                    .neighbors
                    .into_iter()
                    .flatten(),
            );
        }
        for (position, &slot) in created.iter().enumerate() {
            checkpoint(position as u64, self.options(), cancellation)?;
            let entry = self.evaluate(topology, slot)?;
            changed.insert(slot);
            changed.extend(topology.cell(slot).neighbors.into_iter().flatten());
            let key = cell_key(entry.quality.node_identities);
            if (owner.cells.contains_key(&key) && !removed.contains(&key))
                || entries.insert(slot, entry).is_some()
            {
                return Err(error(
                    DelaunayVolumeQualityErrorKind::InvalidQuality,
                    Some(slot as usize),
                    "scoped cavity duplicates retained quality",
                ));
            }
        }
        let update = plan_metric_cap_update(
            self.request().maximum_grading_ratio,
            &Graph {
                owner,
                created: &entries,
                removed: &removed_slots,
                topology,
            },
            changed.iter().copied(),
            || cancellation.is_cancelled(),
        )
        .map_err(update_error)?;
        owner.last_scoped_update = Some((update.invalidated_vertices, update.neighbor_reads));
        tracing::debug!(
            invalidated_vertices = update.invalidated_vertices,
            neighbor_reads = update.neighbor_reads,
            created_cells = created.len(),
            "scoped quality dependency closure planned"
        );
        // Construct all replacement evidence before changing the quality owner.
        // Its surrounding journal supplies atomic nested rollback with topology.
        for (position, (slot, cap)) in update.caps.into_iter().enumerate() {
            checkpoint(position as u64, self.options(), cancellation)?;
            let mut entry = entries
                .remove(&slot)
                .unwrap_or_else(|| owner.cells[&topology.cell_key(slot)].clone());
            let mut raw = self.resolve_raw(&entry.quality.incident_metric_entity_ids)?;
            let dependency = entry.grading.as_mut().unwrap();
            if cap < dependency.raw_length {
                runmat_meshing_size::grading::apply_metric_grading_cap(&mut raw, cap)
                    .map_err(metric_error)?;
            }
            dependency.cap = cap;
            dependency.neighbors = topology.cell(slot).neighbors;
            if !metric_identity(&entry.quality, &raw) {
                entry.quality = self.evaluate_metric(
                    topology,
                    slot,
                    entry.quality.incident_metric_entity_ids.clone(),
                    raw,
                )?;
            }
            entries.insert(slot, entry);
        }
        // Edge endpoints may retain their cap but still need current adjacency.
        for slot in changed {
            if !topology.contains_cell(slot) || entries.contains_key(&slot) {
                continue;
            }
            let key = topology.cell_key(slot);
            let Some(entry) = owner.cells.get(&key) else {
                continue;
            };
            let mut entry = entry.clone();
            entry.grading.as_mut().unwrap().neighbors = topology.cell(slot).neighbors;
            entries.insert(slot, entry);
        }
        for key in removed {
            owner.replace(*key, None);
        }
        for entry in entries.into_values() {
            owner.replace(cell_key(entry.quality.node_identities), Some(entry));
        }
        Ok(())
    }
}
fn metric_identity(
    quality: &DelaunayTetrahedronQuality,
    raw: &runmat_meshing_size::metric::ResolvedMetricEvaluation,
) -> bool {
    [
        quality.resolved_metric.xx,
        quality.resolved_metric.yy,
        quality.resolved_metric.zz,
        quality.resolved_metric.xy,
        quality.resolved_metric.xz,
        quality.resolved_metric.yz,
    ]
    .map(f64::to_bits)
        == [
            raw.metric.xx,
            raw.metric.yy,
            raw.metric.zz,
            raw.metric.xy,
            raw.metric.xz,
            raw.metric.yz,
        ]
        .map(f64::to_bits)
        && quality.active_metric_sources == raw.active_sources
        && quality.applied_metric_contribution_count == raw.applied_contribution_count
        && quality.clipped_metric_contribution_count == raw.clipped_contribution_count
        && quality.rejected_metric_contribution_count == raw.rejected_contribution_count
}
fn update_error(error_value: MetricCapUpdateError) -> DelaunayVolumeQualityError {
    match error_value {
        MetricCapUpdateError::InvalidMetric(failure) => metric_error(failure),
        MetricCapUpdateError::Cancelled => error(
            DelaunayVolumeQualityErrorKind::Cancelled,
            None,
            "scoped metric grading was cancelled",
        ),
    }
}
