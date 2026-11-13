//! Bounded retained search state, tied to one owner's comparison revision.
//! Replay always rechecks current local topology, provenance and quality controls.
use super::*;
mod connectivity;
use connectivity::ConnectivityDelta;

pub(in crate::cdt) struct VolumeSeed {
    pub(in crate::cdt) edits: crate::cdt::mutation::TopologyEdits,
    spectrum: SpectrumDelta,
    connectivity: ConnectivityDelta,
    binding: Arc<()>,
    revision: u64,
    pub(in crate::cdt) last_node_identity: Option<StableDigest>,
}

impl VolumeSeed {
    fn assert_same_baseline(&self, other: &Self) {
        assert!(
            Arc::ptr_eq(&self.binding, &other.binding) && self.revision == other.revision,
            "search comparison requires the same baseline owner and revision"
        );
    }

    pub(in crate::cdt) fn capture(state: &WorkingTrial<'_, '_>) -> Self {
        let edits = state.topology.capture_edits();
        Self {
            connectivity: ConnectivityDelta::from_edits(edits.iter()),
            edits,
            spectrum: state.quality.capture_spectrum(),
            binding: state.binding.clone(),
            revision: state.revision,
            last_node_identity: state
                .topology
                .node_indices
                .last_key_value()
                .map(|(identity, _)| *identity),
        }
    }
    pub(in crate::cdt) fn compare_spectrum(
        &self,
        other: &Self,
        objective: QualityObjective,
    ) -> std::cmp::Ordering {
        self.assert_same_baseline(other);
        self.spectrum.compare(&other.spectrum, objective)
    }
    pub(in crate::cdt) fn compare_joint_spectrum(&self, other: &Self) -> std::cmp::Ordering {
        self.assert_same_baseline(other);
        self.spectrum.compare_joint(&other.spectrum)
    }
    pub(in crate::cdt) fn compare_connectivity(
        &self,
        other: &Self,
        quality: &WorkingVolumeQuality,
    ) -> std::cmp::Ordering {
        self.assert_same_baseline(other);
        self.connectivity.compare(&other.connectivity, quality)
    }
    pub(in crate::cdt) fn connectivity_key(
        &self,
    ) -> (&BTreeSet<[StableDigest; 4]>, &BTreeSet<[StableDigest; 4]>) {
        (&self.connectivity.removed, &self.connectivity.created)
    }
    pub(in crate::cdt) fn replay(
        &self,
        state: &mut WorkingTrial<'_, '_>,
        options: impl Into<VolumeStateOptions>,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), VolumeStateError> {
        let options = options.into();
        if !Arc::ptr_eq(&self.binding, &state.binding) || self.revision != state.revision {
            return Err(error(
                VolumeStateErrorKind::InvalidInput,
                "retained seed belongs to a different search baseline",
            ));
        }
        let applied = state.topology.active_edits();
        if !self.edits.starts_with(applied) {
            return Err(error(
                VolumeStateErrorKind::InvalidInput,
                "retained seed does not extend the active search branch",
            ));
        }
        let applied_len = applied.len();
        for (position, edit) in self.edits.iter().enumerate().skip(applied_len) {
            checkpoint(position as u64, options, cancellation)?;
            match edit {
                TopologyEdit::AppendNode(node) => {
                    state.append_node(*node, options)?;
                }
                TopologyEdit::ReplaceNode {
                    source_identity,
                    replacement,
                    ..
                } => {
                    let index = state.topology.node_index(*source_identity).ok_or_else(|| {
                        error(
                            VolumeStateErrorKind::InvalidTopology,
                            "retained relocation source is absent",
                        )
                    })?;
                    state.replace_node(index, *replacement, options, cancellation)?;
                }
                TopologyEdit::RelocateCavity {
                    replacements,
                    removed,
                    created,
                } => {
                    let replacements = replacements
                        .iter()
                        .map(|(source, replacement)| {
                            state
                                .topology
                                .node_index(*source)
                                .map(|index| (index, *replacement))
                                .ok_or_else(|| {
                                    error(
                                        VolumeStateErrorKind::InvalidTopology,
                                        "retained joint relocation source is absent",
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let removed = removed
                        .iter()
                        .map(|identities| {
                            let vertices = identities
                                .iter()
                                .map(|identity| {
                                    state.topology.node_index(*identity).ok_or_else(|| {
                                        error(
                                            VolumeStateErrorKind::InvalidTopology,
                                            "retained joint cavity corner is absent",
                                        )
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            state
                                .topology
                                .tetrahedron_index(vertices.try_into().expect("four corners"))
                                .ok_or_else(|| {
                                    error(
                                        VolumeStateErrorKind::InvalidTopology,
                                        "retained joint cavity cell is absent",
                                    )
                                })
                        })
                        .collect::<Result<BTreeSet<_>, _>>()?;
                    let created = created
                        .iter()
                        .map(|(identities, region)| {
                            let vertices = identities
                                .iter()
                                .map(|identity| {
                                    if let Some((index, _)) = replacements
                                        .iter()
                                        .find(|(_, node)| node.identity == *identity)
                                    {
                                        Ok(*index)
                                    } else {
                                        state.topology.node_index(*identity).ok_or_else(|| {
                                            error(
                                                VolumeStateErrorKind::InvalidTopology,
                                                "retained joint replacement corner is absent",
                                            )
                                        })
                                    }
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            Ok((vertices.try_into().expect("four corners"), region.clone()))
                        })
                        .collect::<Result<Vec<_>, VolumeStateError>>()?;
                    state.relocate_cavity(
                        crate::cdt::mutation::CavityRelocation {
                            replacements,
                            removed,
                            created,
                        },
                        options,
                        cancellation,
                    )?;
                }
                TopologyEdit::ReplaceCavity { removed, created } => {
                    let nodes = |identities: [StableDigest; 4]| {
                        identities
                            .into_iter()
                            .map(|identity| {
                                state.topology.node_index(identity).ok_or_else(|| {
                                    error(
                                        VolumeStateErrorKind::InvalidTopology,
                                        "retained cavity corner is absent",
                                    )
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map(|vertices| vertices.try_into().expect("four corners"))
                    };
                    let removed = removed
                        .iter()
                        .map(|identities| {
                            let vertices = nodes(*identities)?;
                            state.topology.tetrahedron_index(vertices).ok_or_else(|| {
                                error(
                                    VolumeStateErrorKind::InvalidTopology,
                                    "retained cavity cell is absent",
                                )
                            })
                        })
                        .collect::<Result<BTreeSet<_>, _>>()?;
                    let created = created
                        .iter()
                        .map(|(identities, region)| {
                            nodes(*identities).map(|vertices| (vertices, region.clone()))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    state.replace(&removed, created, options, cancellation)?;
                }
            }
        }
        Ok(())
    }
}
