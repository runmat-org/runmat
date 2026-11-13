//! Exact complete reconstruction comparison without retaining a second report.
//! Independent alternate-arithmetic validation remains a separate final pass.
use super::*;

pub(in crate::cdt) fn verify_complete_handoff(
    topology: &DelaunayVolumeTopology,
    metric_request: &MetricFieldRequest,
    provenance: &DelaunayVolumeProvenance,
    quality: &DelaunayVolumeQuality,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeQualityError> {
    let contexts = validate_inputs(topology, metric_request, provenance, options, cancellation)?;
    if quality.tetrahedra.len() != topology.tetrahedra.len() {
        return Err(mismatch(None));
    }
    visit_from_metric_contexts(
        topology,
        metric_request,
        options,
        contexts,
        stream::BoundaryProtection {
            nodes: &boundary_node_identities(provenance),
            facets: &protected_facet_identities(provenance),
            provenance,
        },
        cancellation,
        |index, reconstructed| {
            if reconstructed != quality.tetrahedra[index] {
                return Err(mismatch(Some(index)));
            }
            Ok(())
        },
    )?;
    if !summary::reduce(topology, &quality.tetrahedra, options, cancellation)?.matches(quality) {
        return Err(mismatch(None));
    }
    validate_delaunay_volume_quality(
        topology,
        metric_request,
        provenance,
        quality,
        options,
        cancellation,
    )
}

fn mismatch(index: Option<usize>) -> DelaunayVolumeQualityError {
    error(
        DelaunayVolumeQualityErrorKind::InvalidQuality,
        index,
        "working quality differs from independent complete handoff evaluation",
    )
}

#[cfg(test)]
mod tests;
