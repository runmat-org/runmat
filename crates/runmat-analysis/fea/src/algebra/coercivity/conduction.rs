//! Structural coercivity of an assembled steady conduction operator, with no global factor.
//!
//! The steady operator is `K = Σₑ Kₑ + Σ_f H_f + Σ_i G_i`: element conduction blocks
//! `Kₑ = ∫ k ∇Nᵀ∇N`, convection face blocks `H_f = ∫ h N Nᵀ` and conductance-interface blocks
//! `G_i = ∫ g (N_a − N_b)(N_a − N_b)ᵀ`. The proof has three parts:
//!
//! 1. **Materials and boundaries.** Every element conductivity `k` is finite and positive, and every
//!    convection coefficient `h` and interface conductance `g` is finite and nonnegative. With the
//!    projection's positive quadrature weights and Jacobians, each `Kₑ` is positive semidefinite
//!    with kernel exactly the constants on that element, and each `H_f` and `G_i` is positive
//!    semidefinite.
//! 2. **Components.** Nodes coupled by a nonzero entry of `K` share an element, a face or an
//!    interface pair, so a temperature `u` with `uᵀKu = 0` is one constant on each connected
//!    component of the operator graph.
//! 3. **Anchors.** Every component holds a prescribed temperature (so its constant is zero), or a
//!    node with positive convection weight `Σ h N_i² dA` (so the constant `c` has energy
//!    `c² ∫ h dA > 0`).
//!
//! Then `K` restricted to the free temperatures is positive definite in exact arithmetic. The
//! assembled values must also be symmetric to the same roundoff-relative tolerance the factor
//! proof uses, with a positive diagonal. The proof gives no eigenvalue bound: the solve is accepted
//! by the normwise backward error of the returned temperature.
use super::*;

/// Why the structural conduction proof refused an operator.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ScalarConductionError {
    #[error("element {element}: thermal conductivity {value} W/(m K) is not finite and positive, so the steady conduction operator is not positive definite")]
    NonPositiveConductivity { element: usize, value: f64 },
    #[error("boundary '{boundary_id}': convection coefficient {value} W/(m² K) is negative or not finite, so the steady conduction operator can be indefinite")]
    NegativeConvection { boundary_id: String, value: f64 },
    #[error("interface '{interface_id}': conductance {value} W/(m² K) is negative or not finite, so the steady conduction operator can be indefinite")]
    NegativeInterfaceConductance { interface_id: String, value: f64 },
    #[error("the steady temperature is undetermined: {nodes} connected nodes (first degree of freedom {first_dof}) have no prescribed temperature and no convection with a positive coefficient; add a temperature or a convection boundary on that body")]
    UnanchoredComponent { nodes: u64, first_dof: usize },
    #[error("degree of freedom {dof}: the steady conduction operator has a nonpositive diagonal {value}")]
    NonPositiveDiagonal { dof: usize, value: f64 },
    #[error("conduction proof inputs disagree: {0}")]
    Shape(&'static str),
}

// Values are reported, never compared numerically; NaN inputs are refused before any compare.
impl Eq for ScalarConductionError {}

/// A completed structural conduction proof for one assembled operator.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarConductionProof {
    dimension: usize,
    components: u64,
    prescribed_components: u64,
    convection_components: u64,
    scalar_operations: u64,
    resident_bytes: u64,
}

/// Bytes per degree of freedom the proof holds: union-find parent (8), component size (8) and
/// anchor flag (1).
const RESIDENT_BYTES_PER_ROW: u64 = 17;

impl ScalarConductionProof {
    /// Proof work: one union per stored entry, a find per entry, and the per-row anchor scan.
    pub(crate) fn operations_for(rows: u64, entries: u64) -> u64 {
        entries
            .saturating_mul(2)
            .saturating_add(rows.saturating_mul(4))
    }

    pub(crate) fn resident_bytes_for(rows: u64) -> u64 {
        rows.saturating_mul(RESIDENT_BYTES_PER_ROW)
    }

    /// The admission of the conduction path: no factor, the proof's inventory and work.
    pub(crate) fn admission(
        support: &crate::algebra::SparsePattern,
    ) -> crate::planning::resource::SolverAdmissionResourcePolicy {
        use crate::planning::resource::{ByteCount, Count};
        let rows = support.rows() as u64;
        let entries = support.column_indices().len() as u64;
        crate::planning::resource::SolverAdmissionResourcePolicy {
            dimension: Count(rows),
            factor_nonzero_upper_bound: Count(0),
            update_count_upper_bound: Count(Self::operations_for(rows, entries)),
            resident_bytes_upper_bound: ByteCount(Self::resident_bytes_for(rows)),
        }
    }

    /// Proves the operator definite on the free temperatures. `stiffness` is the assembled
    /// steady operator before constraints, `prescribed` marks fixed temperatures, and
    /// `positive_convection_weight` holds `Σ h N_i² dA` over faces with `h > 0`.
    pub(crate) fn prove(
        stiffness: &RealSparseMatrix,
        prescribed: &[bool],
        positive_convection_weight: &[f64],
        element_conductivity: &[f64],
    ) -> Result<Self, ScalarConductionError> {
        let n = stiffness.rows();
        if n == 0
            || stiffness.columns() != n
            || prescribed.len() != n
            || positive_convection_weight.len() != n
        {
            return Err(ScalarConductionError::Shape(
                "operator, prescribed mask and convection weights differ in length",
            ));
        }
        if let Some((element, &value)) = element_conductivity
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || **value <= 0.0)
        {
            return Err(ScalarConductionError::NonPositiveConductivity { element, value });
        }
        let mut parent = (0..n).collect::<Vec<usize>>();
        fn find(parent: &mut [usize], mut node: usize) -> usize {
            while parent[node] != node {
                parent[node] = parent[parent[node]];
                node = parent[node];
            }
            node
        }
        let pattern = stiffness.pattern();
        let values = stiffness.values();
        for row in 0..n {
            let mut diagonal = 0.0;
            for entry in pattern.row_range(row) {
                let column = pattern.column(entry);
                if column == row {
                    diagonal = values[entry];
                } else if values[entry] != 0.0 {
                    let (a, b) = (find(&mut parent, row), find(&mut parent, column));
                    if a != b {
                        parent[a.max(b)] = a.min(b);
                    }
                }
            }
            if !diagonal.is_finite() || diagonal <= 0.0 {
                return Err(ScalarConductionError::NonPositiveDiagonal {
                    dof: row,
                    value: diagonal,
                });
            }
        }
        // 0 = unanchored, 1 = prescribed, 2 = convection only.
        let mut anchor = vec![0_u8; n];
        let mut size = vec![0_u64; n];
        for dof in 0..n {
            let root = find(&mut parent, dof);
            size[root] += 1;
            if prescribed[dof] {
                anchor[root] = 1;
            } else if positive_convection_weight[dof] > 0.0 && anchor[root] == 0 {
                anchor[root] = 2;
            }
        }
        let (mut components, mut prescribed_components, mut convection_components) = (0, 0, 0);
        for dof in 0..n {
            if parent[dof] != dof {
                continue;
            }
            components += 1;
            match anchor[dof] {
                1 => prescribed_components += 1,
                2 => convection_components += 1,
                _ => {
                    return Err(ScalarConductionError::UnanchoredComponent {
                        nodes: size[dof],
                        first_dof: dof,
                    })
                }
            }
        }
        Ok(Self {
            dimension: n,
            components,
            prescribed_components,
            convection_components,
            scalar_operations: Self::operations_for(n as u64, values.len() as u64),
            resident_bytes: Self::resident_bytes_for(n as u64),
        })
    }
}

impl CoercivityCertificate {
    /// Binds a conduction proof to the prepared operator: the proof's operator with its
    /// prescribed rows and columns replaced by identity. The prepared values must be symmetric
    /// with a positive diagonal.
    pub(crate) fn from_scalar_conduction(
        matrix: &RealSparseMatrix,
        proof: &ScalarConductionProof,
    ) -> Result<Self, CoercivityError> {
        if matrix.rows() != matrix.columns()
            || matrix.rows() != proof.dimension
            || matrix.properties().symmetry != MatrixSymmetry::Symmetric
        {
            return Err(CoercivityError::BindingMismatch);
        }
        let diagonal = require_symmetric(matrix)?;
        if let Some(pivot) = diagonal
            .iter()
            .position(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(CoercivityError::NonPositivePivot { pivot });
        }
        let mut result = Self {
            pattern_identity: matrix
                .pattern_identity()
                .map_err(|e| CoercivityError::Encoding(e.to_string()))?,
            values_identity: matrix
                .values_identity()
                .map_err(|e| CoercivityError::Encoding(e.to_string()))?,
            dimension: matrix.rows(),
            proof: CoercivityProof::ScalarConduction {
                components: proof.components,
                prescribed_components: proof.prescribed_components,
                convection_components: proof.convection_components,
                scalar_operations: proof.scalar_operations,
                resident_bytes: proof.resident_bytes,
            },
            identity: Digest::sha256([]),
        };
        result.identity = result.expected_identity()?;
        Ok(result)
    }

    /// Whether the structural conduction proof (no eigenvalue bound) certified this operator.
    #[cfg(test)]
    pub(crate) fn proved_by_scalar_conduction(&self) -> bool {
        matches!(self.proof, CoercivityProof::ScalarConduction { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};

    /// A chain of `n` nodes with unit conductance links and `extra` added to each diagonal.
    fn chain(n: usize, links: &[(usize, usize)], extra: &[f64]) -> RealSparseMatrix {
        let mut assembler = RealSparseAssembler::square(
            n,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        );
        for &(a, b) in links {
            assembler.add(a, a, 1.0);
            assembler.add(b, b, 1.0);
            assembler.add(a, b, -1.0);
            assembler.add(b, a, -1.0);
        }
        for (row, &value) in extra.iter().enumerate() {
            assembler.add(row, row, value);
        }
        assembler.finish().unwrap()
    }

    #[test]
    fn prescribed_and_convection_anchors_prove_each_component() {
        // Two bodies: 0-1-2 fixed at node 0, 3-4 cooled by convection at node 4.
        let matrix = chain(5, &[(0, 1), (1, 2), (3, 4)], &[0.0, 0.0, 0.0, 0.0, 0.5]);
        let proof = ScalarConductionProof::prove(
            &matrix,
            &[true, false, false, false, false],
            &[0.0, 0.0, 0.0, 0.0, 0.5],
            &[1.0, 2.0],
        )
        .unwrap();
        assert_eq!(proof.components, 2);
        assert_eq!(proof.prescribed_components, 1);
        assert_eq!(proof.convection_components, 1);
    }

    #[test]
    fn a_floating_body_is_refused_as_undetermined() {
        let matrix = chain(5, &[(0, 1), (1, 2), (3, 4)], &[0.0; 5]);
        let error = ScalarConductionProof::prove(
            &matrix,
            &[true, false, false, false, false],
            &[0.0; 5],
            &[1.0],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ScalarConductionError::UnanchoredComponent {
                nodes: 2,
                first_dof: 3
            }
        );
        assert!(error.to_string().contains("undetermined"));
    }

    #[test]
    fn nonpositive_conductivity_is_refused() {
        let matrix = chain(2, &[(0, 1)], &[0.0; 2]);
        let error = ScalarConductionProof::prove(&matrix, &[true, false], &[0.0; 2], &[1.0, -3.0])
            .unwrap_err();
        assert!(matches!(
            error,
            ScalarConductionError::NonPositiveConductivity { element: 1, .. }
        ));
    }

    #[test]
    fn asymmetric_prepared_values_are_refused() {
        let matrix = chain(2, &[(0, 1)], &[1.0, 0.0]);
        let proof =
            ScalarConductionProof::prove(&matrix, &[false, false], &[1.0, 0.0], &[1.0]).unwrap();
        let mut assembler = RealSparseAssembler::square(
            2,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        );
        for (row, column, value) in [(0, 0, 2.0), (1, 1, 1.0), (0, 1, -1.0), (1, 0, -0.5)] {
            assembler.add(row, column, value);
        }
        let asymmetric = assembler.finish().unwrap();
        assert_eq!(
            CoercivityCertificate::from_scalar_conduction(&asymmetric, &proof).unwrap_err(),
            CoercivityError::AsymmetricValues
        );
        let certificate = CoercivityCertificate::from_scalar_conduction(&matrix, &proof).unwrap();
        assert!(certificate.proved_by_scalar_conduction());
        certificate.validate_for(&matrix).unwrap();
    }
}
