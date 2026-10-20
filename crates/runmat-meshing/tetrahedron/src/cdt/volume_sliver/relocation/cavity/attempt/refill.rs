//! One safeguarded local remesh trial; no background topology is materialized.
use super::*;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use crate::cdt::volume_sliver::working::CaptureSeed;
pub(super) struct RefillTrial<'a, 'b> {
    pub(super) request: &'a CavityAttempt<'b>,
    pub(super) removed: &'a BTreeSet<u32>,
    pub(super) region: &'a Option<runmat_geometry_core::PersistentEntityId>,
    pub(super) boundary: &'a [[u32; 3]],
    pub(super) replacements: Vec<(u32, DelaunayVolumeNode)>,
}
pub(in crate::cdt::volume_sliver::relocation::cavity) struct RefillCandidate {
    pub(in crate::cdt::volume_sliver::relocation::cavity) seed: WorkingSeed,
    pub(in crate::cdt::volume_sliver::relocation::cavity) admitted: bool,
}
impl RefillTrial<'_, '_> {
    pub(super) fn apply<'ctx>(
        self,
        state: &mut impl SearchState<'ctx>,
        work: &mut OptimizationWork<'_>,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Option<RefillCandidate>, DelaunayVolumeSliverError> {
        let Self {
            request,
            removed,
            region,
            boundary,
            replacements,
        } = self;
        let CavityAttempt {
            source: _,
            primary,
            baseline,
            policy,
            prefix,
            ..
        } = *request;
        let topology = state.topology();
        let primary_old = topology.node(primary);
        let replacement = replacements
            .iter()
            .find(|(index, _)| *index == primary)
            .expect("primary replacement")
            .1;
        let relocations = replacements
            .iter()
            .filter(|(index, _)| *index != primary)
            .map(|(index, node)| (topology.node(*index).identity, *node))
            .collect::<Vec<_>>();
        let created = match preserve_vertices(
            topology,
            removed,
            boundary,
            RefillPoint {
                node: replacement,
                replaces: Some(primary_old.identity),
                relocations: &relocations,
            },
            policy.options,
            work,
            cancellation,
        ) {
            Ok(cells) => cells,
            Err(failure) if failure.kind == DelaunayVolumeSliverErrorKind::InvalidTopology => {
                return Ok(None)
            }
            Err(failure) => return Err(failure),
        };
        let created = created
            .into_iter()
            .map(|ids| {
                (
                    ids.map(|id| {
                        replacements
                            .iter()
                            .find(|(_, node)| node.identity == id)
                            .map_or_else(
                                || topology.node_index(id).expect("retained point"),
                                |(index, _)| *index,
                            )
                    }),
                    region.clone(),
                )
            })
            .collect();
        let mut mutations = prefix.to_vec();
        mutations.extend(grouped_relocation_mutations(replacements.iter().map(
            |(index, node)| {
                let cell = topology
                    .ordered_vertex_star(*index)
                    .into_iter()
                    .next()
                    .expect("nonempty star");
                (
                    topology.node(*index).identity,
                    *node,
                    topology
                        .cell(cell)
                        .vertex_indices
                        .map(|v| topology.node(v).identity),
                )
            },
        )));
        let point_count = replacements.len();
        let mut trial = state.trial();
        if let Err(failure) = trial
            .relocate_cavity(
                CavityRelocation {
                    replacements,
                    removed: removed.clone(),
                    created,
                },
                policy.options,
                cancellation,
            )
            .map_err(DelaunayVolumeSliverError::from)
        {
            if failure.kind == DelaunayVolumeSliverErrorKind::InvalidTopology {
                tracing::debug!(reason=%failure,"joint cavity atomic geometry rejected");
                return Ok(None);
            }
            return Err(failure);
        }
        let mut reasons = BTreeMap::new();
        let admitted = crate::cdt::volume_sliver::admission::working_quality_is_admissible(
            trial.quality(),
            baseline,
            policy.refinement,
            policy.objective,
            &mut reasons,
        );
        tracing::debug!(admitted,point_count,reasons=?reasons,"joint cavity cluster quality checked");
        Ok(Some(RefillCandidate {
            seed: trial.capture(mutations),
            admitted,
        }))
    }
}
