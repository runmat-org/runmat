//! One bounded neighboring cluster escapes fixed-star coordinate traps.
//! Final admission uses complete current quality and a grouped atomic edit.
use super::super::working::{CaptureSeed, WorkingSeed};
use super::*;
use crate::cdt::{mutation::CavityRelocation, volume_state::SearchState};

pub(super) fn candidate<'ctx>(
    state: &mut impl SearchState<'ctx>,
    source: &DelaunayTetrahedronQuality,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    prefix: &[DelaunayVolumeSliverMutation],
) -> Result<Option<WorkingSeed>, DelaunayVolumeSliverError> {
    let vertices = super::neighborhood::vertices(state, source);
    if vertices.is_empty() {
        return Ok(None);
    }
    let baseline = state.quality().capture_spectrum();
    // A single prepared cluster replaces repeated pair optimizations. The
    // source and its bounded incidence neighborhood are the only inventory read.
    work.evaluate()?;
    let topology = state.topology();
    let cells = vertices
        .iter()
        .copied()
        .flat_map(|vertex| topology.ordered_vertex_star(vertex))
        .collect::<BTreeSet<_>>();
    let prepared = geometry::block::PreparedBlock::prepare(topology, &vertices, &cells, &|cell| {
        state
            .quality()
            .quality_for_cell(
                topology
                    .cell(cell)
                    .vertex_indices
                    .map(|vertex| topology.node(vertex).identity),
            )
            .expect("current star quality")
            .resolved_metric
    })?;
    let Some(points) = prepared.optimize(cancellation)? else {
        return Ok(None);
    };
    let replacements = vertices
        .into_iter()
        .zip(points)
        .filter_map(|(index, point)| {
            let old = topology.node(index);
            (old.coordinates_m.canonical_value() != point).then(|| {
                (
                    index,
                    DelaunayVolumeNode::explicit(replacement_identity(old.identity), point),
                )
            })
        })
        .collect::<Vec<_>>();
    let mut mutations = prefix.to_vec();
    mutations.extend(grouped_relocation_mutations(replacements.iter().map(
        |(index, replacement)| {
            let source_cell = topology
                .ordered_vertex_star(*index)
                .into_iter()
                .next()
                .expect("nonempty star");
            (
                topology.node(*index).identity,
                *replacement,
                topology
                    .cell(source_cell)
                    .vertex_indices
                    .map(|vertex| topology.node(vertex).identity),
            )
        },
    )));
    let created = cells
        .iter()
        .map(|index| {
            let cell = topology.cell(*index);
            (cell.vertex_indices, cell.region_id.clone())
        })
        .collect();
    let mut trial = state.trial();
    if let Err(failure) = trial
        .relocate_cavity(
            CavityRelocation {
                replacements,
                removed: cells,
                created,
            },
            policy.options,
            cancellation,
        )
        .map_err(DelaunayVolumeSliverError::from)
    {
        if failure.kind == DelaunayVolumeSliverErrorKind::InvalidTopology {
            return Ok(None);
        }
        return Err(failure);
    }
    let mut rejection_counts = BTreeMap::new();
    let admitted = super::super::admission::working_quality_is_admissible(
        trial.quality(),
        &baseline,
        policy.refinement,
        policy.objective,
        &mut rejection_counts,
    );
    tracing::debug!(admitted,reasons=?rejection_counts,"prepared cluster complete quality checked");
    if admitted {
        return Ok(Some(trial.capture(mutations)));
    }
    Ok(None)
}
