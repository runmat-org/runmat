//! Bind constructed local-energy evidence to the actual operator.
use super::*;

impl CoercivityCertificate {
    pub(crate) fn prove_local_energy(
        matrix: &RealSparseMatrix,
        prescribed: &[bool],
        source: crate::algebra::anchored_energy::AnchoredEnergySource,
        normalization: f64,
    ) -> Result<Self, CoercivityError> {
        Self::from_local_energy(matrix, source.prove(matrix, prescribed, normalization)?)
    }

    pub(crate) fn prove_selected_energy(
        matrix: &RealSparseMatrix,
        source: crate::algebra::anchored_energy::AnchoredEnergySource,
        rows: &[Option<usize>],
    ) -> Result<Self, CoercivityError> {
        Self::from_local_energy(matrix, source.prove_selection(matrix, rows)?)
    }

    fn from_local_energy(
        matrix: &RealSparseMatrix,
        verdict: crate::algebra::anchored_energy::EnergyVerdict,
    ) -> Result<Self, CoercivityError> {
        use crate::algebra::anchored_energy::EnergyVerdict;
        let certificate = match verdict {
            EnergyVerdict::Verified(certificate) => certificate,
            EnergyVerdict::Indeterminate {
                graph_lower,
                error_upper,
            } => {
                return Err(CoercivityError::EnergyIndeterminate {
                    graph_lower_bits: graph_lower.to_bits(),
                    error_upper_bits: error_upper.to_bits(),
                })
            }
        };
        if certificate.operator_identity()
            != matrix
                .numerical_identity()
                .map_err(|e| CoercivityError::Encoding(e.to_string()))?
        {
            return Err(CoercivityError::BindingMismatch);
        }
        let mut result = Self {
            pattern_identity: matrix
                .pattern_identity()
                .map_err(|e| CoercivityError::Encoding(e.to_string()))?,
            values_identity: matrix
                .values_identity()
                .map_err(|e| CoercivityError::Encoding(e.to_string()))?,
            dimension: matrix.rows(),
            proof: CoercivityProof::AnchoredLocalEnergy {
                lower_bound_bits: certificate.lower_bound().to_bits(),
                assembly_error_upper_bits: certificate.assembly_error_upper().to_bits(),
                scalar_operations: certificate.scalar_operations(),
                resident_bytes: certificate.resident_bytes(),
            },
            identity: Digest::sha256([]),
        };
        result.identity = result.expected_identity()?;
        Ok(result)
    }
}
