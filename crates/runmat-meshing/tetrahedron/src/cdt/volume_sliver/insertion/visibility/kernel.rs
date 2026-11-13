//! Immutable cavity geometry for repeated candidate evaluation. Expected signs
//! are cached as results, so an unreachable later error stays unreachable.
use super::*;
use runmat_meshing_core::quality::predicate::SpatialPredicateError;
#[cfg(test)]
use runmat_meshing_size::metric::MetricTensor3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SmoothingTrial {
    pub(super) score: f64,
    pub(super) point: [f64; 3],
}

struct BoundaryFace {
    nodes: [DelaunayVolumeNode; 3],
    metric: Option<crate::cdt::volume_quality::PreparedJacobian>,
    expected: Result<PredicateSign, SpatialPredicateError>,
}

pub(super) struct VisibilityKernel {
    faces: Vec<BoundaryFace>,
}
impl VisibilityKernel {
    pub(super) fn prepare(
        topology: &(impl VolumeTopologyRead + ?Sized),
        boundary: &[(u32, usize, [u32; 3])],
        metrics: InsertionMetric<'_>,
    ) -> Result<Self, DelaunayVolumeSliverError> {
        let faces = boundary
            .iter()
            .map(|(cell_index, opposite, face)| {
                let nodes = face.map(|vertex| topology.node(vertex));
                let interior = topology.node(topology.cell(*cell_index).vertex_indices[*opposite]);
                Ok(BoundaryFace {
                    nodes,
                    metric: crate::cdt::volume_quality::PreparedJacobian::new(metrics(*face)?)
                        .map_err(|cause| {
                            error(
                                DelaunayVolumeSliverErrorKind::Accounting(cause),
                                "visibility metric numerical accounting failed",
                            )
                        })?,
                    expected: orient_volume_nodes([nodes[0], nodes[1], nodes[2], interior]),
                })
            })
            .collect::<Result<_, DelaunayVolumeSliverError>>()?;
        Ok(Self { faces })
    }
    pub(super) fn contains(&self, point: [f64; 3]) -> Result<bool, DelaunayVolumeSliverError> {
        let probe = DelaunayVolumeNode::explicit(kernel_probe_identity(point), point);
        for face in &self.faces {
            let expected = face.expected.map_err(visibility_predicate_error)?;
            let [a, b, c] = face.nodes;
            let observed =
                orient_volume_nodes([a, b, c, probe]).map_err(visibility_predicate_error)?;
            if observed == PredicateSign::Zero || observed != expected {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(super) fn score(&self, point: [f64; 3]) -> Result<f64, DelaunayVolumeSliverError> {
        self.faces
            .iter()
            .try_fold(f64::INFINITY, |minimum, face| {
                let [a, b, c] = face.nodes.map(|node| node.coordinates_m.canonical_value());
                let score = match face.metric.as_ref() {
                    Some(metric) => metric
                        .score([a, b, c, point])
                        .map_err(|cause| {
                            error(
                                DelaunayVolumeSliverErrorKind::Accounting(cause),
                                "visibility score numerical accounting failed",
                            )
                        })?
                        .unwrap_or(0.),
                    None => 0.0,
                };
                Ok(minimum.min(score))
            })
            .map(|minimum| if minimum.is_infinite() { 0.0 } else { minimum })
    }
    pub(super) fn best_trial(
        &self,
        point: [f64; 3],
        radius: f64,
        best_score: f64,
    ) -> Result<Option<SmoothingTrial>, DelaunayVolumeSliverError> {
        let mut candidates = Vec::with_capacity(26);
        for x in -1_i8..=1 {
            for y in -1_i8..=1 {
                for z in -1_i8..=1 {
                    if x == 0 && y == 0 && z == 0 {
                        continue;
                    }
                    let norm = f64::from(x * x + y * y + z * z).sqrt();
                    candidates.push(std::array::from_fn(|axis| {
                        point[axis] + radius * f64::from([x, y, z][axis]) / norm
                    }));
                }
            }
        }
        let usage = runmat_simulation_usage::UsageContext::capture();
        let results = runmat_cpu_execution::ordered_map(
            &candidates,
            |&candidate| -> Result<Option<SmoothingTrial>, DelaunayVolumeSliverError> {
                let _usage = usage
                    .as_ref()
                    .map(runmat_simulation_usage::UsageContext::enter);
                if !self.contains(candidate)? {
                    return Ok(None);
                }
                let score = self.score(candidate)?;
                Ok((score > best_score).then_some(SmoothingTrial {
                    score,
                    point: candidate,
                }))
            },
        );
        let mut best = None::<SmoothingTrial>;
        // Canonical input order also preserves the original first error.
        for result in results {
            let Some(candidate) = result? else {
                continue;
            };
            if best.as_ref().is_none_or(|previous| {
                candidate.score > previous.score
                    || candidate.score == previous.score
                        && candidate.point.map(f64::to_bits) < previous.point.map(f64::to_bits)
            }) {
                best = Some(candidate);
            }
        }
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn node(identity: u8, point: [f64; 3]) -> DelaunayVolumeNode {
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([identity; 32]), point)
    }
    #[test]
    fn cached_later_error_remains_unreachable_after_earlier_rejection() {
        let nodes = [
            node(1, [0., 0., 0.]),
            node(2, [0., 1., 0.]),
            node(3, [0., 0., 1.]),
        ];
        let expected = orient_volume_nodes([nodes[0], nodes[1], nodes[2], node(4, [1., 0., 0.])]);
        let failure = SpatialPredicateError::NonFiniteCoordinate {
            point_index: 0,
            axis: 0,
        };
        let metric = crate::cdt::volume_quality::PreparedJacobian::new(
            MetricTensor3::isotropic_length_m(1.).unwrap(),
        )
        .unwrap();
        let kernel = VisibilityKernel {
            faces: vec![
                BoundaryFace {
                    nodes,
                    expected,
                    metric,
                },
                BoundaryFace {
                    nodes,
                    metric,
                    expected: Err(failure),
                },
            ],
        };
        assert_eq!(kernel.contains([-1., 0.1, 0.1]), Ok(false));
        assert_eq!(
            kernel.contains([1., 0.1, 0.1]),
            Err(visibility_predicate_error(failure))
        );
        for threads in [1, 4] {
            assert_eq!(
                runmat_cpu_execution::with_cpu_pool(threads, || kernel.best_trial(
                    [-1., 0.1, 0.1],
                    0.,
                    0.
                ))
                .unwrap(),
                Ok(None)
            );
            assert_eq!(
                runmat_cpu_execution::with_cpu_pool(threads, || kernel.best_trial(
                    [1., 0.1, 0.1],
                    0.,
                    0.
                ))
                .unwrap(),
                Err(visibility_predicate_error(failure))
            );
        }
    }
}
