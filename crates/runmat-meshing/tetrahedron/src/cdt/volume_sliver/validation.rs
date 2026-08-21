use std::collections::BTreeSet;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    error, run_treatment, validate_input, validate_options, DelaunayVolumeSliverError,
    DelaunayVolumeSliverErrorKind, DelaunayVolumeSliverOptions, DelaunayVolumeSliverTreatment,
};
use crate::cdt::DelaunayVolumeRefinementInput;

pub fn validate_delaunay_volume_sliver_treatment(
    input: DelaunayVolumeRefinementInput<'_>,
    treatment: &DelaunayVolumeSliverTreatment,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeSliverError> {
    validate_options(options)?;
    validate_input(input, options, cancellation)?;
    if treatment.relocations.len() as u64 > options.maximum_passes
        || treatment.relocations.iter().any(|relocation| {
            relocation.source_node_identity == StableDigest::ZERO
                || relocation.replacement_node.identity == StableDigest::ZERO
                || relocation.source_node_identity == relocation.replacement_node.identity
                || relocation
                    .replacement_node
                    .coordinates_m
                    .iter()
                    .any(|value| !value.is_finite())
        })
    {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "relocation evidence is malformed or exceeds its pass limit",
        ));
    }
    let replacement_identities = treatment
        .relocations
        .iter()
        .map(|relocation| relocation.replacement_node.identity)
        .collect::<BTreeSet<_>>();
    if replacement_identities.len() != treatment.relocations.len() {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "relocation replacement identities must be unique",
        ));
    }
    if treatment.quality.tetrahedra.iter().any(|tetrahedron| {
        tetrahedron.metric_scaled_jacobian < input.quality_options.minimum_metric_scaled_jacobian
    }) {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidQuality,
            "completed sliver treatment retains a metric scaled-Jacobian violation",
        ));
    }
    let expected = run_treatment(input, options, cancellation)?;
    if expected != *treatment {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidInput,
            "sliver treatment does not match deterministic checked replay",
        ));
    }
    Ok(())
}
