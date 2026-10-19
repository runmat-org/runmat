//! Candidate point geometry is local and immutable. Resolve each prospective
//! child's actual provenance scope once and reuse it through all probes.
use super::*;
use crate::cdt::volume_quality::PreparedJacobian;

struct Child {
    face: [[f64; 3]; 3],
    metric: Option<PreparedJacobian>,
}

pub(in crate::cdt::volume_sliver::insertion) struct PreparedEdgeStar {
    children: Vec<Child>,
}
impl PreparedEdgeStar {
    pub(in crate::cdt::volume_sliver::insertion) fn prepare(
        topology: &(impl VolumeTopologyRead + ?Sized),
        star: &[u32],
        edge: [u32; 2],
        metrics: InsertionMetric<'_>,
    ) -> Result<Self, DelaunayVolumeSliverError> {
        let mut children = Vec::with_capacity(2 * star.len());
        for cell in star {
            let vertices = topology.cell(*cell).vertex_indices;
            let mut opposite = vertices.into_iter().filter(|vertex| !edge.contains(vertex));
            let first = opposite.next().expect("edge star corner");
            let second = opposite.next().expect("edge star corner");
            for endpoint in edge {
                let face = [endpoint, first, second];
                children.push(Child {
                    face: face.map(|vertex| topology.node(vertex).coordinates_m),
                    metric: PreparedJacobian::new(metrics(face)?),
                });
            }
        }
        Ok(Self { children })
    }

    pub(in crate::cdt::volume_sliver::insertion) fn score(&self, point: [f64; 3]) -> f64 {
        self.children
            .iter()
            .map(|child| {
                let [a, b, c] = child.face;
                child
                    .metric
                    .as_ref()
                    .and_then(|metric| metric.score([a, b, c, point]))
                    .unwrap_or(0.)
            })
            .min_by(f64::total_cmp)
            .unwrap_or(0.)
    }
}

#[cfg(test)]
mod tests;
