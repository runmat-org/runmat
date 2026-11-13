use crate::cdt::DelaunayVolumeProvenance;

/// Merge in place so normalization does not retain a second complete binding array.
pub(super) fn normalize(mut provenance: DelaunayVolumeProvenance) -> DelaunayVolumeProvenance {
    provenance
        .nodes
        .sort_by_key(|binding| binding.node_identity);
    provenance.nodes.dedup_by(|duplicate, retained| {
        if retained.node_identity != duplicate.node_identity {
            return false;
        }
        retained.entity_ids.append(&mut duplicate.entity_ids);
        retained.entity_ids.sort();
        retained.entity_ids.dedup();
        true
    });
    provenance
        .segments
        .sort_by_key(|binding| binding.node_identities);
    provenance.segments.dedup();
    provenance
        .facets
        .sort_by_key(|binding| binding.node_identities);
    provenance.facets.dedup_by(|duplicate, retained| {
        if retained.node_identities != duplicate.node_identities
            || retained.chart_id != duplicate.chart_id
            || retained.entity_ids != duplicate.entity_ids
        {
            return false;
        }
        retained.region_ids.append(&mut duplicate.region_ids);
        retained.region_ids.sort();
        retained.region_ids.dedup();
        true
    });
    provenance
}
