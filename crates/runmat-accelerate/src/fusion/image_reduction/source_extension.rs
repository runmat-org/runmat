//! Extend a detected image group only through actual selected publication
//! casts. A source store is not an arithmetic permission by itself.

use std::collections::HashSet;

use crate::fusion::{detect_fusion_groups, FusionGroup, FusionKind, FusionPattern};
use crate::graph::{AccelGraph, NodeId, ValueOrigin};

use super::{publication_follows_stage, ImageSourcePublication};

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
    let mut selected = detected_base(graph, base)?;
    let mut nodes = selected.nodes.iter().copied().collect::<HashSet<_>>();
    for publication in publications {
        for node in cast_path(graph, publication)? {
            let cast = graph.node(node)?;
            if nodes.insert(node) {
                // Existing interior casts are part of the detected source;
                // only a suffix after its final producer may be added.
                if cast.span.start <= base.span.end {
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
        let Some(FusionPattern::ImageNormalize(pattern)) = base.pattern.as_ref() else {
            return false;
        };
        let Some(final_source) = pattern
            .reduction_routes
            .as_ref()
            .and_then(|routes| routes.stage_producers())
            .map(|producers| producers.output)
        else {
            return false;
        };
        let base_nodes = base.nodes.iter().copied().collect::<HashSet<_>>();
        let extra = selected
            .nodes
            .iter()
            .copied()
            .filter(|node| !base_nodes.contains(node))
            .collect::<Vec<_>>();
        if selected.nodes.len() != base.nodes.len() + extra.len()
            || !base.nodes.iter().all(|node| selected.nodes.contains(node))
        {
            return false;
        }
        let mut current = final_source;
        for node in &extra {
            let Some(cast) = graph.node(*node) else {
                return false;
            };
            if cast.span.start <= base.span.end
                || cast.span.end > selected.span.end
                || !super::super::is_canonical_single_cast(cast)
                || cast.inputs.as_slice() != [current]
            {
                return false;
            }
            let [output] = cast.outputs.as_slice() else {
                return false;
            };
            current = *output;
        }
        selected.span.end
            == extra
                .last()
                .and_then(|node| graph.node(*node))
                .map_or(base.span.end, |node| node.span.end)
    })
}
