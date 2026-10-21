//! Fluid mesh motion: the interior fluid nodes follow the wall by a volume-weighted Laplace
//! extension of the interface displacement. Small cells are stiffer, so the wall's boundary
//! layer cells translate with it instead of collapsing.
use super::FluidStructureError as Error;
use crate::{
    algebra::{
        spmv, ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties,
        OperatorRole, RealSparseAssembler, SparsePattern,
    },
    topology::{AnalysisProjection, FiniteVolumeTopology},
};

pub(super) struct MeshMotion {
    pub free_nodes: Vec<usize>,
    /// Interior coupling to prescribed nodes, by free row: (prescribed node, coefficient).
    pub prescribed_coupling: Vec<Vec<(usize, f64)>>,
    pub operator: super::SpdSystem,
}

impl MeshMotion {
    /// `prescribed[node]` marks the fluid nodes whose motion is given: every node on a fluid
    /// boundary face, which includes the interface nodes carried by the structure.
    pub(super) fn assemble(
        projection: &AnalysisProjection,
        fluid: &FiniteVolumeTopology,
        prescribed: &[bool],
    ) -> Result<Self, Error> {
        let mut free_ordinal = vec![None; projection.nodes.len()];
        let mut free_nodes = Vec::new();
        for cell in &fluid.cells {
            for &node in &projection.elements[cell.source_element_index].node_indices {
                if !prescribed[node] && free_ordinal[node].is_none() {
                    free_ordinal[node] = Some(free_nodes.len());
                    free_nodes.push(node);
                }
            }
        }
        let n = free_nodes.len();
        let mut rows = vec![Vec::<usize>::new(); n];
        let mut coupling = vec![Vec::<(usize, f64)>::new(); n];
        let mut entries = Vec::new();
        for cell in &fluid.cells {
            let element = &projection.elements[cell.source_element_index];
            let nodes = &element.node_indices;
            let m = nodes.len();
            let mut local = vec![0.0; m * m];
            let weight = 1.0 / cell.volume_m3;
            projection
                .visit_element_quadrature(element, |point| {
                    let grads = point.physical_shape_gradients();
                    for i in 0..m {
                        for j in 0..m {
                            local[i * m + j] += weight
                                * (grads[i][0] * grads[j][0]
                                    + grads[i][1] * grads[j][1]
                                    + grads[i][2] * grads[j][2])
                                * point.volume_weight_m3;
                        }
                    }
                })
                .map_err(|e| Error::InvalidDetail(e.to_string()))?;
            for i in 0..m {
                let Some(row) = free_ordinal[nodes[i]] else {
                    continue;
                };
                for j in 0..m {
                    let value = local[i * m + j];
                    match free_ordinal[nodes[j]] {
                        Some(column) => {
                            rows[row].push(column);
                            entries.push((row, column, value));
                        }
                        None => coupling[row].push((nodes[j], value)),
                    }
                }
            }
        }
        let mut offsets = vec![0u64];
        let mut columns = Vec::new();
        for row in &mut rows {
            row.sort_unstable();
            row.dedup();
            columns.extend(row.iter().map(|&c| c as u64));
            offsets.push(columns.len() as u64);
        }
        let pattern = SparsePattern::try_new(n as u64, n as u64, offsets, columns)
            .map_err(|e| Error::Operator(e.to_string()))?;
        let mut assembler = RealSparseAssembler::with_pattern(
            pattern,
            OperatorRole::EffectiveTangent,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .map_err(|e| Error::Operator(e.to_string()))?;
        assembler.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Fluid,
            contribution_id: ContributionId("fluid-mesh-motion-laplacian".into()),
        });
        for (row, column, value) in entries {
            assembler.add(row, column, value);
        }
        let operator = assembler
            .finish()
            .map_err(|e| Error::Operator(e.to_string()))?;
        Ok(Self {
            free_nodes,
            prescribed_coupling: coupling,
            operator: super::SpdSystem::new(operator),
        })
    }

    /// Interior displacement for the given prescribed node displacement (per component).
    pub(super) fn extend(
        &self,
        prescribed: &[[f64; 3]],
        policy: super::LinearPolicy,
    ) -> Result<Vec<[f64; 3]>, Error> {
        let mut result = prescribed.to_vec();
        if self.free_nodes.is_empty() {
            return Ok(result);
        }
        for k in 0..3 {
            let rhs = self
                .prescribed_coupling
                .iter()
                .map(|row| {
                    -row.iter()
                        .map(|&(node, a)| a * prescribed[node][k])
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            if rhs.iter().all(|v| *v == 0.0) {
                for &node in &self.free_nodes {
                    result[node][k] = 0.0;
                }
                continue;
            }
            let (x, _) = self.operator.solve(&rhs, policy)?;
            for (&node, value) in self.free_nodes.iter().zip(x) {
                result[node][k] = value;
            }
        }
        Ok(result)
    }

    /// Relative residual of the Laplace extension for a retained displacement field.
    pub(super) fn residual(&self, displacement: &[[f64; 3]]) -> Result<f64, Error> {
        let mut numerator = 0.0f64;
        let mut denominator = 0.0f64;
        for k in 0..3 {
            let x = self
                .free_nodes
                .iter()
                .map(|&node| displacement[node][k])
                .collect::<Vec<_>>();
            let action =
                spmv(self.operator.matrix(), &x).map_err(|e| Error::Operator(e.to_string()))?;
            for (row, a) in action.iter().enumerate() {
                let b = -self.prescribed_coupling[row]
                    .iter()
                    .map(|&(node, c)| c * displacement[node][k])
                    .sum::<f64>();
                numerator = numerator.hypot(a - b);
                denominator = denominator.hypot(b);
            }
        }
        Ok(super::relative(numerator, denominator))
    }
}
