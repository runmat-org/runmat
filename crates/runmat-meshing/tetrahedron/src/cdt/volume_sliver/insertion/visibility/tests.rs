//! Compare complete trajectories with the original serial evaluator, not just
//! final candidate scores. No equilibrium or acceptance controls are changed.
use super::*;
use runmat_meshing_core::NeverCancelled;
use runmat_meshing_size::metric::MetricTensor3;

// witness: FEA-SHARED-RESOURCES-001/machine_cpu_execution @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Four starting points with isotropic and anisotropic metrics preserve the original
//   serial smoothing trajectory bit for bit at one, two and four admitted CPU threads, including
//   successful searches. This covers prepared cavity geometry and ordered trial selection, not
//   arbitrary meshing providers, whole-study throughput, memory accounting or distributed
//   execution.
#[test]
fn prepared_parallel_smoothing_preserves_complete_serial_trajectory() {
    let (topology, _) = crate::cdt::volume_sliver::tests::interior_sliver_fixture();
    let source = &topology.tetrahedra[0];
    let neighbor = source.neighbors.iter().flatten().next().copied().unwrap();
    let protected = BTreeSet::new();
    let mut improved = 0;
    for start in [
        [0.15, 0.2, 0.2],
        [0.25, 0.25, 0.25],
        [0.3, 0.1, 0.1],
        [0.1, 0.3, 0.3],
    ] {
        for metric in [
            MetricTensor3::isotropic_length_m(1.).unwrap(),
            MetricTensor3 {
                xx: 4.,
                yy: 1.,
                zz: 0.25,
                xy: 0.,
                xz: 0.,
                yz: 0.,
            },
        ] {
            let metrics = |_| Ok(metric);
            let context = || VisibilityContext {
                source_vertices: source.vertex_indices,
                neighbor_index: neighbor,
                protected_faces: &protected,
                metrics: &metrics,
                target_scaled_jacobian: 0.,
                options: DelaunayVolumeSliverOptions::default(),
                cancellation: &NeverCancelled,
            };
            let expected =
                reference_smooth_visibility_cavity_point(&topology, start, context()).unwrap();
            improved += usize::from(expected.is_some());
            for threads in [1, 2, 4] {
                let actual = runmat_cpu_execution::with_cpu_pool(threads, || {
                    smooth_visibility_cavity_point(&topology, start, context())
                })
                .unwrap()
                .unwrap();
                assert_eq!(
                    actual.map(|point| point.map(f64::to_bits)),
                    expected.map(|point| point.map(f64::to_bits))
                );
            }
        }
    }
    assert!(
        improved > 0,
        "matrix must execute a successful smoothing search"
    );
}

fn reference_smooth_visibility_cavity_point(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    start: [f64; 3],
    context: VisibilityContext<'_>,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let VisibilityContext {
        source_vertices,
        neighbor_index,
        protected_faces,
        metrics,
        target_scaled_jacobian,
        options,
        cancellation,
    } = context;
    let source_index = topology
        .tetrahedron_index(source_vertices)
        .expect("quality source remains in topology");
    let probe = |point| DelaunayVolumeNode::explicit(kernel_probe_identity(point), point);
    let Ok((removed, _)) = grow_visibility_cavity(
        topology,
        VisibilityCavityRequest {
            relocated_vertices: &[],
            node: probe(start),
            source_index,
            neighbor_index,
            protected_faces,
            quality_growth: Some((metrics, target_scaled_jacobian)),
            options,
            cancellation,
        },
    )?
    else {
        return Ok(None);
    };
    let boundary = cavity_boundary_faces(topology, &removed);
    let score = |point: [f64; 3]| {
        boundary
            .iter()
            .filter_map(|(_, _, face)| {
                let points = [
                    topology.node(face[0]).coordinates_m.canonical_value(),
                    topology.node(face[1]).coordinates_m.canonical_value(),
                    topology.node(face[2]).coordinates_m.canonical_value(),
                    point,
                ];
                crate::cdt::volume_quality::evaluate_tetrahedron_geometry(
                    points,
                    metrics(*face).unwrap(),
                )
                .unwrap()
                .map(|quality| quality.metric_scaled_jacobian)
            })
            .min_by(f64::total_cmp)
            .unwrap_or(0.0)
    };
    let in_kernel = |point: [f64; 3]| -> Result<bool, DelaunayVolumeSliverError> {
        for (tetrahedron_index, opposite, face) in &boundary {
            let tetrahedron = &topology.cell(*tetrahedron_index);
            let nodes = face.map(|vertex| topology.node(vertex));
            let interior = topology.node(tetrahedron.vertex_indices[*opposite]);
            let expected = orient_volume_nodes([nodes[0], nodes[1], nodes[2], interior])
                .map_err(visibility_predicate_error)?;
            let observed = orient_volume_nodes([nodes[0], nodes[1], nodes[2], probe(point)])
                .map_err(visibility_predicate_error)?;
            if observed == PredicateSign::Zero || observed != expected {
                return Ok(false);
            }
        }
        Ok(true)
    };
    let cavity_vertices = removed
        .iter()
        .flat_map(|index| topology.cell(*index).vertex_indices)
        .collect::<BTreeSet<_>>();
    let points = cavity_vertices
        .iter()
        .map(|vertex| topology.node(*vertex).coordinates_m.canonical_value())
        .collect::<Vec<_>>();
    let spans = std::array::from_fn::<_, 3, _>(|axis| {
        points
            .iter()
            .map(|point| point[axis])
            .fold(f64::NEG_INFINITY, f64::max)
            - points
                .iter()
                .map(|point| point[axis])
                .fold(f64::INFINITY, f64::min)
    });
    let local_scale = spans.into_iter().fold(0., f64::hypot);
    if local_scale == 0.0 {
        return Ok(None);
    }
    let mut point = start;
    let initial_score = score(point);
    let mut best_score = initial_score;
    let mut radius = local_scale * 0.25;
    let minimum_radius = local_scale * 2_f64.powi(-24);
    for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
        if iteration.is_multiple_of(8) && cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "sliver cavity smoothing was cancelled",
            ));
        }
        let mut best = None;
        for x in -1_i8..=1 {
            for y in -1_i8..=1 {
                for z in -1_i8..=1 {
                    if x == 0 && y == 0 && z == 0 {
                        continue;
                    }
                    let norm = f64::from(x * x + y * y + z * z).sqrt();
                    let candidate = std::array::from_fn(|axis| {
                        point[axis] + radius * f64::from([x, y, z][axis]) / norm
                    });
                    if !in_kernel(candidate)? {
                        continue;
                    }
                    let candidate_score = score(candidate);
                    if candidate_score > best_score
                        && best
                            .as_ref()
                            .is_none_or(|(score, point): &(f64, [f64; 3])| {
                                candidate_score > *score
                                    || candidate_score == *score
                                        && candidate.map(f64::to_bits) < point.map(f64::to_bits)
                            })
                    {
                        best = Some((candidate_score, candidate));
                    }
                }
            }
        }
        if let Some((score, candidate)) = best {
            best_score = score;
            point = candidate;
        } else {
            radius *= 0.5;
            if radius < minimum_radius {
                break;
            }
        }
    }
    Ok((best_score > initial_score).then_some(point))
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A four-cell cavity considers other poor interior faces even when its worst face is
//   protected. The resulting cavity retains exactly the original outer facets. This expansion check
//   does not guarantee arbitrary protected geometries admit a quality mesh.
#[test]
fn protected_worst_face_does_not_hide_other_quality_expansion_faces() {
    let (topology, provenance) = crate::cdt::volume_sliver::tests::interior_sliver_fixture();
    let protected = provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect();
    let source_index = 0;
    let neighbor_index = topology
        .cell(source_index)
        .neighbors
        .into_iter()
        .flatten()
        .next()
        .unwrap();
    let node = DelaunayVolumeNode::explicit(StableDigest::from_bytes([240; 32]), [0.3, 0.3, 1e-5]);
    let metric = MetricTensor3::isotropic_length_m(1.).unwrap();
    let (removed, _) = grow_visibility_cavity(
        &topology,
        VisibilityCavityRequest {
            relocated_vertices: &[],
            node,
            source_index,
            neighbor_index,
            protected_faces: &protected,
            quality_growth: Some((&|_| Ok(metric), 1.)),
            options: DelaunayVolumeSliverOptions::default(),
            cancellation: &NeverCancelled,
        },
    )
    .unwrap()
    .expect("quality growth must preserve the immutable outer boundary");
    assert_eq!(removed.len(), topology.tetrahedra.len());
    let boundary = cavity_boundary_faces(&topology, &removed);
    assert_eq!(boundary.len(), protected.len());
    assert!(
        boundary
            .iter()
            .all(|(_, _, face)| protected
                .contains(&face.map(|vertex| topology.node(vertex).identity)))
    );
}
