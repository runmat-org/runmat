use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId};

use super::super::{
    error, DelaunayVolumeMesh, DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind,
    DelaunayVolumeMeshStage,
};
use crate::cdt::{DelaunayConstraintFacetSide, DelaunayConstraints};

pub(super) fn validate_exact_sources(
    exact: &ExactBRepTopology,
    constraints: &DelaunayConstraints,
    mesh: &DelaunayVolumeMesh,
) -> Result<(), DelaunayVolumeMeshError> {
    let vertices = constraints
        .nodes
        .iter()
        .filter_map(|node| {
            node.source_vertex_id
                .clone()
                .map(|source| (source, node.identity))
        })
        .collect::<BTreeMap<_, _>>();
    for binding in &mesh.provenance.nodes {
        if binding.entity_ids.len() != 1
            || vertices.get(&binding.entity_ids[0]) != Some(&binding.node_identity)
        {
            return Err(invalid(
                "node provenance does not match an admitted exact vertex",
            ));
        }
    }

    let edges = exact
        .edges
        .iter()
        .map(|edge| edge.id.clone())
        .collect::<BTreeSet<_>>();
    for binding in &mesh.provenance.segments {
        if binding.entity_ids.len() != 1 || !edges.contains(&binding.entity_ids[0]) {
            return Err(invalid("segment provenance names an unadmitted exact edge"));
        }
        let admitted = constraints.segments.iter().any(|segment| {
            segment.source_edge_id.as_ref() == binding.entity_ids.first()
                && segment.source_edge_parameters.is_some_and(|parameters| {
                    let lower = parameters[0].min(parameters[1]);
                    let upper = parameters[0].max(parameters[1]);
                    binding
                        .edge_parameters
                        .iter()
                        .all(|parameter| *parameter >= lower && *parameter <= upper)
                })
        });
        if !admitted {
            return Err(invalid(
                "segment provenance parameters are outside the admitted exact edge interval",
            ));
        }
    }

    let mut entity_charts = BTreeMap::<PersistentEntityId, BTreeSet<_>>::new();
    let mut regions = BTreeSet::new();
    for facet in &constraints.facets {
        entity_charts
            .entry(facet.source_face_id.clone())
            .or_default()
            .insert(facet.chart_id);
        for contact in &facet.contact_ids {
            entity_charts
                .entry(contact.clone())
                .or_default()
                .insert(facet.chart_id);
        }
        collect_region(&facet.positive_side, &mut regions);
        collect_region(&facet.negative_side, &mut regions);
    }
    for binding in &mesh.provenance.facets {
        if binding.entity_ids.iter().any(|entity| {
            !entity_charts
                .get(entity)
                .is_some_and(|charts| charts.contains(&binding.chart_id))
        }) || binding
            .region_ids
            .iter()
            .any(|region| !regions.contains(region))
        {
            return Err(invalid(
                "facet provenance does not match an admitted exact face, contact, chart, or region",
            ));
        }
    }
    if mesh.topology.tetrahedra.iter().any(|tetrahedron| {
        tetrahedron
            .region_id
            .as_ref()
            .is_none_or(|region| !regions.contains(region))
    }) {
        return Err(invalid(
            "volume topology contains an unassigned or unadmitted exact region",
        ));
    }
    Ok(())
}

fn collect_region(side: &DelaunayConstraintFacetSide, regions: &mut BTreeSet<PersistentEntityId>) {
    if let DelaunayConstraintFacetSide::Region(region) = side {
        regions.insert(region.clone());
    }
}

fn invalid(reason: impl Into<String>) -> DelaunayVolumeMeshError {
    error(
        DelaunayVolumeMeshStage::Provenance,
        DelaunayVolumeMeshErrorKind::InvalidGeometry,
        reason,
    )
}
