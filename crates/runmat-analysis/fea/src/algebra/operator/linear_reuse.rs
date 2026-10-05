//! One bounded process-local proof, rebound only to its exact constrained operator.
use super::*;
use crate::algebra::CoercivityCertificate;

#[derive(Default)]
pub(crate) struct LinearPreparationReuse {
    certificate: Option<CoercivityCertificate>,
}

impl LinearPreparationReuse {
    pub(crate) const RESIDENT_BYTES: u64 = std::mem::size_of::<Self>() as u64;

    pub(crate) fn clear(&mut self) {
        self.certificate = None;
    }

    pub(super) fn prepare(
        &mut self,
        matrix: &RealSparseMatrix,
        rhs: &[f64],
        source: Option<&NullspaceEliminationEvidence>,
    ) -> Result<PreparedSolverAdmissibility, CoercivityError> {
        if let Some(certificate) = self
            .certificate
            .as_ref()
            .filter(|certificate| certificate.validate_for(matrix).is_ok())
        {
            return PreparedSolverAdmissibility::bind_existing_with_source(
                certificate.clone(),
                matrix,
                rhs,
                source,
            );
        }
        // A failed replacement cannot leave a stale proof available for later work.
        self.clear();
        let evidence = PreparedSolverAdmissibility::prove(matrix, rhs, source)?;
        self.certificate = Some(evidence.coercivity().clone());
        Ok(evidence)
    }
}

#[cfg(test)]
mod tests;
