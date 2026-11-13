//! Extend a detected image group only through actual selected publication
//! casts. A source store is not an arithmetic permission by itself.

use std::collections::HashSet;

use crate::fusion::{detect_fusion_groups, FusionGroup, FusionKind, FusionPattern};
use crate::graph::{AccelGraph, NodeId, ValueId, ValueOrigin};

use super::{publication_follows_stage, ImageSourcePublication};

fn checkpoint_producers(group: &FusionGroup) -> Option<[Option<ValueId>; 5]> {
    let FusionPattern::ImageNormalize(pattern) = group.pattern.as_ref()? else {
        return None;
    };
    let producers = pattern.reduction_routes.as_ref()?.stage_producers()?;
    Some([
        Some(producers.mu),
        Some(producers.sigma),
        producers.pre_clamp_output,
        producers.clamp_output,
        Some(producers.output),
    ])
}

fn same_detected_group(left: &FusionGroup, right: &FusionGroup) -> bool {
    left.id == right.id
        && left.kind == right.kind
        && left.nodes == right.nodes
        && left.shape == right.shape
        && left.span == right.span
        && matches!(
            (&left.pattern, &right.pattern),
            (Some(FusionPattern::ImageNormalize(a)), Some(FusionPattern::ImageNormalize(b)))
                if super::source_invocation::same_image_pattern(a, b)
        )
}

fn detected_base(graph: &AccelGraph, selected: &FusionGroup) -> Option<FusionGroup> {
    detect_fusion_groups(graph)
        .into_iter()
        .find(|actual| same_detected_group(actual, selected))
}

fn cast_path(graph: &AccelGraph, publication: &ImageSourcePublication) -> Option<Vec<NodeId>> {
    if !publication_follows_stage(
        graph,
        publication.producer,
        publication.published,
        publication.statement.clone(),
    ) {
        return None;
    }
    let mut current = publication.published;
    let mut path = Vec::new();
    let mut visited = HashSet::new();
    while current != publication.producer {
        let ValueOrigin::NodeOutput { node, output } = &graph.value(current)?.origin else {
            return None;
        };
        let cast = graph.node(*node)?;
        if !visited.insert(*node)
            || cast.outputs.get(*output) != Some(&current)
            || !super::super::is_canonical_single_cast(cast)
        {
            return None;
        }
        path.push(*node);
        current = *cast.inputs.first()?;
    }
    path.reverse();
    Some(path)
}

/// Reconstruct the detected base and add only canonical source `single`
/// conversions proven by the selected VM publications. The caller still
/// has to bind actual values and prove numerical/physical stage equivalence.
pub fn selected_group_with_publications(
    graph: &AccelGraph,
    base: &FusionGroup,
    publications: &[ImageSourcePublication],
) -> Option<FusionGroup> {
    if base.kind != FusionKind::ImageNormalize || publications.len() < 3 {
        return None;
    }
    let checkpoints = checkpoint_producers(base)?;
    if publications.len() != checkpoints.iter().flatten().count()
        || publications
            .iter()
            .zip(checkpoints.into_iter().flatten())
            .any(|(publication, checkpoint)| publication.producer != checkpoint)
    {
        return None;
    }
    let mut selected = detected_base(graph, base)?;
    let mut nodes = selected.nodes.iter().copied().collect::<HashSet<_>>();
    for publication in publications {
        for node in cast_path(graph, publication)? {
            let cast = graph.node(node)?;
            if nodes.insert(node) {
                // The exact VM publication authorizes this canonical cast at
                // its source statement. Publication casts for mean and sigma
                // can be interleaved with the detected arithmetic; they are
                // not necessarily a suffix after the final image producer.
                if cast.span.start < publication.statement.start
                    || cast.span.end >= publication.statement.end
                {
                    return None;
                }
                selected.nodes.push(node);
                selected.span.end = selected.span.end.max(cast.span.end);
            }
        }
    }
    selected.nodes.sort_by_key(|id| {
        graph
            .node(*id)
            .map(|node| (node.span.start, node.span.end, *id))
    });
    Some(selected)
}

pub(super) fn matches_extended_group(graph: &AccelGraph, selected: &FusionGroup) -> bool {
    detect_fusion_groups(graph).iter().any(|base| {
        if base.kind != FusionKind::ImageNormalize
            || base.id != selected.id
            || base.shape != selected.shape
            || !matches!(
                (&base.pattern, &selected.pattern),
                (Some(FusionPattern::ImageNormalize(a)), Some(FusionPattern::ImageNormalize(b)))
                    if super::source_invocation::same_image_pattern(a, b)
            )
            || selected.span.start != base.span.start
            || selected.span.end < base.span.end
        {
            return false;
        }
        let Some(checkpoints) = checkpoint_producers(base) else {
            return false;
        };
        let base_nodes = base.nodes.iter().copied().collect::<HashSet<_>>();
        let extra = selected
            .nodes
            .iter()
            .copied()
            .filter(|node| !base_nodes.contains(node))
            .collect::<Vec<_>>();
        let selected_nodes = selected.nodes.iter().copied().collect::<HashSet<_>>();
        if selected_nodes.len() != selected.nodes.len()
            || selected.nodes.len() != base.nodes.len() + extra.len()
            || !base.nodes.iter().all(|node| selected.nodes.contains(node))
        {
            return false;
        }
        // Added nodes must form canonical cast paths rooted at exact semantic
        // image checkpoints. This accepts interleaved mean/sigma publication
        // casts without admitting unrelated arithmetic into the image group.
        let mut reachable = checkpoints.into_iter().flatten().collect::<HashSet<_>>();
        for node in &extra {
            let Some(cast) = graph.node(*node) else {
                return false;
            };
            let [input] = cast.inputs.as_slice() else {
                return false;
            };
            let [output] = cast.outputs.as_slice() else {
                return false;
            };
            if cast.span.end > selected.span.end
                || !super::super::is_canonical_single_cast(cast)
                || !reachable.contains(input)
            {
                return false;
            }
            reachable.insert(*output);
        }
        selected.span.end
            == extra
                .iter()
                .filter_map(|node| graph.node(*node))
                .fold(base.span.end, |end, node| end.max(node.span.end))
    })
}
