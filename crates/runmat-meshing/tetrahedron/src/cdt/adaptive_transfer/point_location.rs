use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    MeshingCancellationSignal, SolverEntityTransfer, SolverMeshArtifact, SolverTransferMethod,
    SolverTransferSource, StableDigest,
};

use super::{
    error, invalid_artifact, DelaunayAdaptiveTransferError, DelaunayAdaptiveTransferErrorKind,
    DelaunayAdaptiveTransferOptions,
};

pub(super) fn build_volume_element_transfers(
    source: &SolverMeshArtifact,
    target: &SolverMeshArtifact,
    options: DelaunayAdaptiveTransferOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<SolverEntityTransfer>, DelaunayAdaptiveTransferError> {
    let source_identities = source
        .topology
        .volume_elements
        .iter()
        .map(|element| element.stable_identity)
        .collect::<BTreeSet<_>>();
    let mut work = PointLocationWork::new(options, cancellation);
    let mut transfers = Vec::new();
    for target_element in &target.topology.volume_elements {
        if source_identities.contains(&target_element.stable_identity) {
            continue;
        }
        let centroid = element_centroid(target, target_element)?;
        let source_identity = locate_source_element(source, centroid, &mut work)?;
        transfers.push(SolverEntityTransfer {
            target_stable_identity: target_element.stable_identity,
            method: SolverTransferMethod::CentroidProjection,
            sources: vec![SolverTransferSource {
                stable_identity: source_identity,
                weight: 1.0,
            }],
        });
    }
    transfers.sort_by_key(|transfer| transfer.target_stable_identity);
    Ok(transfers)
}

fn element_centroid(
    artifact: &SolverMeshArtifact,
    element: &runmat_meshing_core::SolverVolumeElement,
) -> Result<[f64; 3], DelaunayAdaptiveTransferError> {
    let nodes = artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut centroid = [0.0; 3];
    for node in &element.node_ids[..4] {
        let point = nodes
            .get(node)
            .ok_or_else(|| invalid_artifact("solver element references a missing node"))?;
        for axis in 0..3 {
            centroid[axis] += point[axis] * 0.25;
        }
    }
    Ok(centroid)
}

fn locate_source_element(
    source: &SolverMeshArtifact,
    point: [f64; 3],
    work: &mut PointLocationWork<'_>,
) -> Result<StableDigest, DelaunayAdaptiveTransferError> {
    let nodes = source
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut containing = Vec::new();
    for element in &source.topology.volume_elements {
        let mut tetrahedron = [[0.0; 3]; 4];
        for (index, node) in element.node_ids[..4].iter().enumerate() {
            tetrahedron[index] = *nodes
                .get(node)
                .ok_or_else(|| invalid_artifact("solver element references a missing node"))?;
        }
        let mut inside = true;
        for replace in 0..4 {
            work.predicate()?;
            let mut points = tetrahedron;
            points[replace] = point;
            if !matches!(
                orient3d(points).map_err(|failure| {
                    error(
                        DelaunayAdaptiveTransferErrorKind::ProjectionFailure,
                        format!("exact point location rejected coordinates: {failure:?}"),
                    )
                })?,
                PredicateSign::Positive | PredicateSign::Zero
            ) {
                inside = false;
                break;
            }
        }
        if inside {
            containing.push(element.stable_identity);
        }
    }
    containing.into_iter().min().ok_or_else(|| {
        error(
            DelaunayAdaptiveTransferErrorKind::ProjectionFailure,
            "target element centroid is outside the admitted source mesh",
        )
    })
}

struct PointLocationWork<'a> {
    predicates: u64,
    options: DelaunayAdaptiveTransferOptions,
    cancellation: &'a dyn MeshingCancellationSignal,
}

impl<'a> PointLocationWork<'a> {
    fn new(
        options: DelaunayAdaptiveTransferOptions,
        cancellation: &'a dyn MeshingCancellationSignal,
    ) -> Self {
        Self {
            predicates: 0,
            options,
            cancellation,
        }
    }

    fn predicate(&mut self) -> Result<(), DelaunayAdaptiveTransferError> {
        if self
            .predicates
            .is_multiple_of(self.options.cancellation_check_interval)
            && self.cancellation.is_cancelled()
        {
            return Err(error(
                DelaunayAdaptiveTransferErrorKind::Cancelled,
                "cancelled",
            ));
        }
        if self.predicates >= self.options.maximum_point_location_predicates {
            return Err(error(
                DelaunayAdaptiveTransferErrorKind::ResourceLimit,
                "adaptive transfer point-location predicate limit exceeded",
            ));
        }
        self.predicates += 1;
        Ok(())
    }
}
