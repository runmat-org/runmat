//! Accepted local topology and quality with nested rollback and revision-bound
//! edit seeds. Algorithm policies and semantic mutation evidence live with their
//! refinement/shape owners, not in this shared state.
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use crate::cdt::mutation::{
    CavityTrial, TopologyEdit, TopologyMutation, TopologyMutationError, TopologyMutationErrorKind,
};
use crate::cdt::topology::VolumeTopologyRead;
use crate::cdt::volume_quality::{
    working::{PreparedLocalQuality, QualityTrial, SpectrumDelta, WorkingVolumeQuality},
    QualityObjective,
};
use crate::cdt::{
    DelaunayInsertionOptions, DelaunayVolumeNode, DelaunayVolumeQuality,
    DelaunayVolumeQualityOptions, DelaunayVolumeTopology,
};
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use std::{collections::BTreeSet, sync::Arc};
mod commit;
mod error;
mod insertion;
mod relocation;
mod seed;
use error::{error, insertion_error, quality_error};
pub(in crate::cdt) use error::{VolumeStateError, VolumeStateErrorKind};
pub(in crate::cdt) use seed::VolumeSeed;

#[derive(Clone, Copy)]
pub(in crate::cdt) struct VolumeStateOptions {
    pub(in crate::cdt) insertion: DelaunayInsertionOptions,
    pub(in crate::cdt) cancellation_check_interval: u64,
}
fn checkpoint(
    index: u64,
    options: VolumeStateOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), VolumeStateError> {
    if options.cancellation_check_interval == 0 {
        return Err(error(
            VolumeStateErrorKind::InvalidOptions,
            "working-state cancellation interval must be nonzero",
        ));
    }
    if index.is_multiple_of(options.cancellation_check_interval) && cancellation.is_cancelled() {
        return Err(error(
            VolumeStateErrorKind::Cancelled,
            "working-state operation cancelled",
        ));
    }
    Ok(())
}

pub(in crate::cdt) struct WorkingState<'ctx> {
    protected_nodes: BTreeSet<StableDigest>,
    prepared: &'ctx PreparedLocalQuality<'ctx>,
    topology: TopologyMutation,
    quality: WorkingVolumeQuality,
    binding: Arc<()>,
    revision: u64,
}

pub(in crate::cdt) struct WorkingTrial<'a, 'ctx> {
    protected_nodes: &'a BTreeSet<StableDigest>,
    prepared: &'ctx PreparedLocalQuality<'ctx>,
    topology: CavityTrial<'a>,
    quality: QualityTrial<'a>,
    binding: Arc<()>,
    revision: u64,
}

pub(in crate::cdt) struct CommittedVolumeChange {
    binding: Arc<()>,
    revision: u64,
}

pub(in crate::cdt) trait SearchState<'ctx> {
    fn preparation(&self) -> &'ctx PreparedLocalQuality<'ctx>;
    fn protected_node(&self, identity: StableDigest) -> bool;
    fn quality_options(&self) -> DelaunayVolumeQualityOptions;
    fn topology(&self) -> &TopologyMutation;
    fn quality(&self) -> &WorkingVolumeQuality;
    fn trial(&mut self) -> WorkingTrial<'_, 'ctx>;
}

pub(in crate::cdt) struct DiagnosticVolumeState {
    pub(in crate::cdt) topology: DelaunayVolumeTopology,
    pub(in crate::cdt) quality: DelaunayVolumeQuality,
    pub(in crate::cdt) verification: Result<(), VolumeStateError>,
}

impl<'ctx> WorkingState<'ctx> {
    pub(in crate::cdt) fn new(
        topology: DelaunayVolumeTopology,
        quality: DelaunayVolumeQuality,
        options: DelaunayVolumeQualityOptions,
        prepared: &'ctx PreparedLocalQuality<'ctx>,
        fixed_nodes: &[StableDigest],
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, VolumeStateError> {
        #[cfg(test)]
        prepared.observe_lifecycle(true);
        let span = tracing::info_span!(
            "meshing.tetrahedron.working_state_prepare",
            node_count = topology.nodes.len(),
            tetrahedron_count = topology.tetrahedra.len()
        );
        let _guard = span.enter();
        if options != prepared.options() {
            return Err(error(
                VolumeStateErrorKind::InvalidOptions,
                "working quality and preparation controls differ",
            ));
        }
        let mut topology = TopologyMutation::from_canonical(topology);
        let mut quality = WorkingVolumeQuality::new(quality, &topology, options, cancellation)
            .map_err(quality_error)?;
        prepared
            .initialize_scoped(&mut quality, &topology, cancellation)
            .map_err(quality_error)?;
        if quality.cell_count() != topology.active_tetrahedron_count() {
            return Err(error(
                VolumeStateErrorKind::InvalidInput,
                "working topology and quality inventories differ",
            ));
        }
        // Prepare shared roots outside speculative trials. Rejected candidates
        // retain those roots rather than rebuilding the full index each time.
        let mut poll = || {
            if cancellation.is_cancelled() {
                Err(error(
                    VolumeStateErrorKind::Cancelled,
                    "working search preparation was cancelled",
                ))
            } else {
                Ok(())
            }
        };
        topology.prepare_point_spatial_index(&mut poll)?;
        topology.prepare_cell_spatial_index(&mut poll)?;
        topology.enable_edit_trace();
        let provenance = prepared.provenance();
        let mut protected_nodes = provenance
            .facets
            .iter()
            .flat_map(|facet| facet.node_identities)
            .chain(
                provenance
                    .segments
                    .iter()
                    .flat_map(|segment| segment.node_identities),
            )
            .chain(provenance.nodes.iter().map(|node| node.node_identity))
            .collect::<BTreeSet<_>>();
        protected_nodes.extend(fixed_nodes.iter().copied());
        Ok(Self {
            protected_nodes,
            prepared,
            topology,
            quality,
            binding: Arc::new(()),
            revision: 0,
        })
    }
    #[cfg(test)]
    pub(in crate::cdt) fn rebase(
        &mut self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), VolumeStateError> {
        let mut poll = || {
            if cancellation.is_cancelled() {
                Err(error(
                    VolumeStateErrorKind::Cancelled,
                    "working search rebase was cancelled",
                ))
            } else {
                Ok(())
            }
        };
        // Flush accepted changes before snapshotting a speculative cache. The
        // journal contains the current local edit, never prior accepted passes.
        self.topology.prepare_point_spatial_index(&mut poll)?;
        self.topology.prepare_cell_spatial_index(&mut poll)?;
        self.finish_rebase();
        Ok(())
    }
    pub(in crate::cdt) fn finish(
        self,
        options: impl Into<VolumeStateOptions>,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(DelaunayVolumeTopology, DelaunayVolumeQuality), VolumeStateError> {
        let diagnostic = self.into_diagnostic(options, cancellation)?;
        diagnostic.verification?;
        Ok((diagnostic.topology, diagnostic.quality))
    }
    /// Preserve a rejected state for explicit diagnosis even when its cached
    /// quality disagrees with independent reconstruction. This is no mesh token.
    pub(in crate::cdt) fn into_diagnostic(
        self,
        options: impl Into<VolumeStateOptions>,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<DiagnosticVolumeState, VolumeStateError> {
        let options = options.into();
        #[cfg(test)]
        self.prepared.observe_lifecycle(false);
        let span = tracing::info_span!(
            "meshing.tetrahedron.canonical_handoff",
            tetrahedron_count = self.topology.active_tetrahedron_count()
        );
        let _guard = span.enter();
        let Self {
            prepared,
            topology,
            quality,
            protected_nodes,
            binding,
            revision: _,
        } = self;
        drop(protected_nodes);
        drop(binding);
        let quality_options = prepared.options();
        let quality = quality
            .into_cells(quality_options, cancellation)
            .map_err(quality_error)?;
        let topology = topology
            .freeze(options.insertion.topology, cancellation)
            .map_err(mutation_error)?;
        let quality = quality
            .summarize(&topology, quality_options, cancellation)
            .map_err(quality_error)?;
        let verification = crate::cdt::volume_quality::verify_complete_handoff(
            &topology,
            prepared.request(),
            prepared.provenance(),
            &quality,
            quality_options,
            cancellation,
        )
        .map_err(quality_error);
        Ok(DiagnosticVolumeState {
            topology,
            quality,
            verification,
        })
    }
}

impl<'ctx> SearchState<'ctx> for WorkingState<'ctx> {
    fn preparation(&self) -> &'ctx PreparedLocalQuality<'ctx> {
        self.prepared
    }
    fn protected_node(&self, identity: StableDigest) -> bool {
        self.protected_nodes.contains(&identity)
    }
    fn quality_options(&self) -> DelaunayVolumeQualityOptions {
        self.prepared.options()
    }
    fn topology(&self) -> &TopologyMutation {
        &self.topology
    }
    fn quality(&self) -> &WorkingVolumeQuality {
        &self.quality
    }
    fn trial(&mut self) -> WorkingTrial<'_, 'ctx> {
        WorkingTrial {
            protected_nodes: &self.protected_nodes,
            prepared: self.prepared,
            topology: self.topology.cavity_trial(),
            quality: self.quality.trial(),
            binding: self.binding.clone(),
            revision: self.revision,
        }
    }
}
impl<'ctx> SearchState<'ctx> for WorkingTrial<'_, 'ctx> {
    fn preparation(&self) -> &'ctx PreparedLocalQuality<'ctx> {
        self.prepared
    }
    fn protected_node(&self, identity: StableDigest) -> bool {
        self.protected_nodes.contains(&identity)
    }
    fn quality_options(&self) -> DelaunayVolumeQualityOptions {
        self.prepared.options()
    }
    fn topology(&self) -> &TopologyMutation {
        &self.topology
    }
    fn quality(&self) -> &WorkingVolumeQuality {
        &self.quality
    }
    fn trial(&mut self) -> WorkingTrial<'_, 'ctx> {
        WorkingTrial {
            protected_nodes: self.protected_nodes,
            prepared: self.prepared,
            topology: self.topology.cavity_trial(),
            quality: self.quality.trial(),
            binding: self.binding.clone(),
            revision: self.revision,
        }
    }
}

impl WorkingTrial<'_, '_> {
    pub(in crate::cdt) fn replace_node(
        &mut self,
        index: u32,
        replacement: DelaunayVolumeNode,
        options: impl Into<VolumeStateOptions>,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), VolumeStateError> {
        let options = options.into();
        let removed = self.topology.vertex_stars[index as usize]
            .iter()
            .map(|cell| self.topology.cell_key(*cell))
            .collect::<Vec<_>>();
        let cells = self
            .topology
            .replace_node(
                index,
                replacement,
                self.protected_nodes,
                options.insertion.topology,
                cancellation,
            )
            .map_err(mutation_error)?;
        self.prepared
            .replace_quality(
                &mut self.quality,
                &removed,
                &cells,
                &*self.topology,
                cancellation,
            )
            .map_err(quality_error)
    }
    pub(in crate::cdt) fn commit(self) {
        self.topology.commit();
        self.quality.commit();
    }
    pub(in crate::cdt) fn append_node(
        &mut self,
        node: DelaunayVolumeNode,
        options: impl Into<VolumeStateOptions>,
    ) -> Result<u32, VolumeStateError> {
        let options = options.into();
        if self.topology.nodes.len() as u64 >= options.insertion.topology.maximum_nodes {
            return Err(error(
                VolumeStateErrorKind::ResourceLimit,
                "local insertion exceeds current node inventory limit",
            ));
        }
        self.topology.append_node(node).map_err(mutation_error)
    }
    pub(in crate::cdt) fn replace(
        &mut self,
        removed: &BTreeSet<u32>,
        created: Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>,
        options: impl Into<VolumeStateOptions>,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), VolumeStateError> {
        let options = options.into();
        let removed_keys = removed
            .iter()
            .map(|index| self.topology.cell_key(*index))
            .collect::<Vec<_>>();
        let created = self
            .topology
            .replace_closed_cavity(
                removed,
                created,
                self.prepared.protected_facets(),
                options.insertion.topology,
                cancellation,
            )
            .map_err(mutation_error)?;
        self.prepared
            .replace_quality(
                &mut self.quality,
                &removed_keys,
                &created,
                &*self.topology,
                cancellation,
            )
            .map_err(quality_error)
    }
    pub(in crate::cdt) fn capture_edits(&self) -> VolumeSeed {
        VolumeSeed::capture(self)
    }
}

fn mutation_error(failure: TopologyMutationError) -> VolumeStateError {
    let kind = match failure.kind {
        TopologyMutationErrorKind::ResourceLimit => VolumeStateErrorKind::ResourceLimit,
        TopologyMutationErrorKind::Cancelled => VolumeStateErrorKind::Cancelled,
        TopologyMutationErrorKind::InvalidTopology => VolumeStateErrorKind::InvalidTopology,
    };
    error(kind, failure.to_string())
}
