//! Geometry analyses on the grid test kernel (`GeometryAnalyzer`).

use super::*;

impl crate::evaluate::GeometryAnalyzer for FakeKernel {
    /// Mass properties exactly from grid occupancy (orthogonal cells), draft from face normals,
    /// and the rest from the grid and its faces ([`super::analysis`]).
    fn analyze(
        &self,
        request: &crate::evaluate::AnalysisRequest,
        bodies: &[&BodyHandle],
    ) -> Result<crate::evaluate::AnalysisResult, KernelError> {
        use crate::evaluate::{AnalysisRequest as R, AnalysisResult, MassProperties};
        match request {
            R::MassProperties { density } => {
                let mut v = 0.0;
                let mut m1 = [0.0; 3];
                // Second moments about the origin: ∫ x_i x_j dV.
                let mut m2 = [[0.0; 3]; 3];
                for (k, h) in bodies.iter().enumerate() {
                    let s = h
                        .downcast::<FakeSolid>()
                        .ok_or(KernelError::ForeignOperand { operand: k as u32 })?;
                    let d = s.dims();
                    for z in 0..d[2] {
                        for y in 0..d[1] {
                            for x in 0..d[0] {
                                if !s.occ[x + d[0] * (y + d[1] * z)] {
                                    continue;
                                }
                                let c = [x, y, z];
                                let lo: Vec<f64> = (0..3).map(|a| s.grid[a][c[a]]).collect();
                                let hi: Vec<f64> = (0..3).map(|a| s.grid[a][c[a] + 1]).collect();
                                let w: Vec<f64> = (0..3).map(|a| hi[a] - lo[a]).collect();
                                let dv = w[0] * w[1] * w[2];
                                v += dv;
                                let mid: Vec<f64> = (0..3).map(|a| (lo[a] + hi[a]) / 2.0).collect();
                                for i in 0..3 {
                                    m1[i] += dv * mid[i];
                                    for j in 0..3 {
                                        m2[i][j] += if i == j {
                                            dv * (mid[i] * mid[i] + w[i] * w[i] / 12.0)
                                        } else {
                                            dv * mid[i] * mid[j]
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
                if v <= 0.0 {
                    return Err(KernelError::Failed {
                        message: "no volume".into(),
                    });
                }
                let c = [m1[0] / v, m1[1] / v, m1[2] / v];
                // Central second moments, then the inertia tensor.
                let mut cm = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        cm[i][j] = m2[i][j] - v * c[i] * c[j];
                    }
                }
                let tr = cm[0][0] + cm[1][1] + cm[2][2];
                let mut inertia = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        inertia[i][j] = density * (if i == j { tr } else { 0.0 } - cm[i][j]);
                    }
                }
                Ok(AnalysisResult::MassProperties(MassProperties {
                    volume: v,
                    mass: v * density,
                    centroid: c,
                    inertia,
                }))
            }
            R::Draft { pull, min } => {
                let first = bodies.first().ok_or(KernelError::Failed {
                    message: "no body".into(),
                })?;
                let s = first
                    .downcast::<FakeSolid>()
                    .ok_or(KernelError::ForeignOperand { operand: 0 })?;
                Ok(AnalysisResult::Draft(crate::evaluate::draft_analysis(
                    &s.topology,
                    *pull,
                    *min,
                )))
            }
            R::SectionProperties { origin, normal } => {
                super::analysis::section(bodies, *origin, *normal)
            }
            R::Thickness { min, max } => super::analysis::thickness(bodies, *min, *max),
            R::Curvature => super::analysis::curvature(bodies),
            R::Zebra => super::analysis::zebra(bodies),
            R::Deviation => super::analysis::deviation(bodies),
            R::Undercut { pull } => super::analysis::undercut(bodies, *pull),
        }
    }
}
