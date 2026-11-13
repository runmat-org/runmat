//! Meshing-owned affine Tet4/Tet10 point interpolation with shared predicate admission.
use super::point_location::PointLocationWork;
use super::{
    error, DelaunayAdaptiveTransferError as Error, DelaunayAdaptiveTransferErrorKind as Kind,
};
use runmat_meshing_core::{
    interpolation::tetrahedron10_lagrange_weights,
    quality::predicate::{orient3d, PredicateSign},
    ElementOrder, SolverEntityTransfer, SolverMeshArtifact, SolverTransferMethod,
    SolverTransferSource, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};
use std::collections::{BTreeMap, BTreeSet};

fn determinant(p: [[f64; 3]; 4]) -> f64 {
    let a = p[0];
    let b = std::array::from_fn::<_, 3, _>(|i| p[1][i] - a[i]);
    let c = std::array::from_fn::<_, 3, _>(|i| p[2][i] - a[i]);
    let d = std::array::from_fn::<_, 3, _>(|i| p[3][i] - a[i]);
    b[0] * (c[1] * d[2] - c[2] * d[1]) - b[1] * (c[0] * d[2] - c[2] * d[0])
        + b[2] * (c[0] * d[1] - c[1] * d[0])
}

pub(super) fn build_node_transfers(
    source: &SolverMeshArtifact,
    target: &SolverMeshArtifact,
    work: &mut PointLocationWork<'_>,
) -> Result<Vec<SolverEntityTransfer>, Error> {
    let source_nodes = source
        .topology
        .nodes
        .iter()
        .map(|n| (n.node_id, n))
        .collect::<BTreeMap<_, _>>();
    let identities = source
        .topology
        .nodes
        .iter()
        .map(|n| n.stable_identity)
        .collect::<BTreeSet<_>>();
    let mut elements = source.topology.volume_elements.iter().collect::<Vec<_>>();
    elements.sort_by_key(|e| e.stable_identity);
    for element in &elements {
        if element.order == ElementOrder::Tet10 {
            let p: [[f64; 3]; 4] = std::array::from_fn(|i| {
                source_nodes[&element.node_ids[i]]
                    .coordinates_m
                    .canonical_value()
            });
            let scale = (1..4)
                .flat_map(|i| (0..3).map(move |a| (p[i][a] - p[0][a]).abs()))
                .fold(0.0_f64, f64::max);
            if !scale.is_finite() || scale == 0.0 {
                return Err(error(
                    Kind::ProjectionFailure,
                    "source affine scale is not representable",
                ));
            }
            for (i, [a, b]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
                work.predicate()?;
                let mid = source_nodes[&element.node_ids[4 + i]]
                    .coordinates_m
                    .canonical_value();
                if (0..3).any(|axis| {
                    (mid[axis] - 0.5 * (p[a][axis] + p[b][axis])).abs()
                        > 128.0 * f64::EPSILON * scale
                }) {
                    return Err(error(
                        Kind::UnsupportedGeometry,
                        "adaptive nodal transfer requires affine source Tet10 geometry",
                    ));
                }
            }
        }
    }
    let mut transfers = Vec::new();
    for node in &target.topology.nodes {
        if identities.contains(&node.stable_identity) {
            continue;
        }
        let mut found = None;
        for element in &elements {
            let p: [[f64; 3]; 4] = std::array::from_fn(|i| {
                source_nodes[&element.node_ids[i]]
                    .coordinates_m
                    .canonical_value()
            });
            // Artifact validation owns exact orientation. This floating triple
            // product has the opposite sign to orient3d; barycentric ratios
            // preserve that common sign rather than imposing another convention.
            let volume = determinant(p);
            if !volume.is_finite() || volume == 0.0 {
                return Err(error(
                    Kind::ProjectionFailure,
                    "unresolved affine source determinant",
                ));
            }
            let mut bary = [0.0; 4];
            let mut inside = true;
            for (i, value) in bary.iter_mut().enumerate() {
                work.predicate()?;
                let mut sample = p;
                sample[i] = node.coordinates_m.canonical_value();
                match orient3d(sample).map_err(|e| {
                    error(
                        Kind::ProjectionFailure,
                        format!("point predicate failed: {e:?}"),
                    )
                })? {
                    PredicateSign::Negative => {
                        inside = false;
                        break;
                    }
                    PredicateSign::Zero => *value = 0.0,
                    PredicateSign::Positive => {
                        *value = determinant(sample) / volume;
                        if !value.is_finite() || *value <= 0.0 {
                            return Err(error(Kind::ProjectionFailure,"source coordinates cannot resolve an admitted positive barycentric weight"));
                        }
                    }
                }
            }
            if !inside {
                continue;
            }
            let sum = bary.iter().sum::<f64>();
            if !sum.is_finite() || (sum - 1.0).abs() > 128.0 * f64::EPSILON {
                return Err(error(
                    Kind::ProjectionFailure,
                    "affine point coordinates do not preserve partition of unity",
                ));
            }
            bary = bary.map(|v| v / sum);
            let (weights, method) = if element.order == ElementOrder::Tet10 {
                (
                    tetrahedron10_lagrange_weights(bary).to_vec(),
                    SolverTransferMethod::QuadraticLagrangeInterpolation,
                )
            } else {
                (
                    bary.to_vec(),
                    SolverTransferMethod::BarycentricInterpolation,
                )
            };
            let mut sources = element
                .node_ids
                .iter()
                .zip(weights)
                .filter_map(|(id, weight)| {
                    (weight != 0.0).then_some(SolverTransferSource {
                        stable_identity: source_nodes[id].stable_identity,
                        weight,
                    })
                })
                .collect::<Vec<_>>();
            sources.sort_by_key(|s| s.stable_identity);
            found = Some(SolverEntityTransfer {
                target_stable_identity: node.stable_identity,
                method,
                sources,
            });
            break;
        }
        transfers.push(found.ok_or_else(|| {
            error(
                Kind::ProjectionFailure,
                "new target node is outside admitted source mesh",
            )
        })?);
    }
    transfers.sort_by_key(|t| t.target_stable_identity);
    Ok(transfers)
}
