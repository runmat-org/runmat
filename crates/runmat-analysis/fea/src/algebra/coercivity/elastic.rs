//! Bind a completed elastic rigidity proof to the operator it certifies.
use super::*;
use crate::algebra::elastic_rigidity::{
    ElasticRigidityProof, ANCHOR_MINIMUM_RANK_RATIO, ELASTIC_ELEMENT_MARGIN_ROUNDING_FACTOR,
};

impl CoercivityCertificate {
    /// `matrix` is either the full operator with the proof's constrained rows decoupled, or the
    /// operator reduced to exactly the proof's free DOFs.
    pub(crate) fn from_elastic_rigidity(
        matrix: &RealSparseMatrix,
        proof: &ElasticRigidityProof,
    ) -> Result<Self, CoercivityError> {
        let free = proof.fixed().iter().filter(|fixed| !**fixed).count();
        if matrix.rows() != matrix.columns()
            || (matrix.rows() != proof.fixed().len() && matrix.rows() != free)
            || matrix.properties().symmetry != MatrixSymmetry::Symmetric
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
            proof: CoercivityProof::ElasticRigidity {
                elements: proof.elements(),
                components: proof.components(),
                element_margin_rounding_factor_bits: ELASTIC_ELEMENT_MARGIN_ROUNDING_FACTOR
                    .to_bits(),
                anchor_rank_ratio_bits: ANCHOR_MINIMUM_RANK_RATIO.to_bits(),
                scalar_operations: proof.scalar_operations(),
                resident_bytes: proof.resident_bytes(),
            },
            identity: Digest::sha256([]),
        };
        result.identity = result.expected_identity()?;
        Ok(result)
    }

    /// Whether the elastic rigidity proof (no eigenvalue bound) certified this operator.
    pub(crate) fn proved_by_elastic_rigidity(&self) -> bool {
        matches!(self.proof, CoercivityProof::ElasticRigidity { .. })
    }
}
