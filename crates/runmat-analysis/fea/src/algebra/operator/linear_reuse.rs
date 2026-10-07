//! One bounded process-local proof, rebound only to its exact constrained operator.
use super::*;
use crate::algebra::CoercivityCertificate;

#[derive(Default)]
pub(crate) struct LinearPreparationReuse {
    certificate: Option<CoercivityCertificate>,
    local_energy: Option<(
        crate::algebra::anchored_energy::AnchoredEnergySource,
        Vec<bool>,
        f64,
    )>,
}

impl LinearPreparationReuse {
    pub(crate) const RESIDENT_BYTES: u64 = std::mem::size_of::<Self>() as u64;

    pub(crate) fn clear(&mut self) {
        self.certificate = None;
        self.local_energy = None;
    }

    /// Original source is consumed once and freed after proof; no serialized claim enters.
    pub(crate) fn with_local_energy(
        source: crate::algebra::anchored_energy::AnchoredEnergySource,
        prescribed: Vec<bool>,
        normalization: f64,
    ) -> Self {
        Self {
            certificate: None,
            local_energy: Some((source, prescribed, normalization)),
        }
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
        let local_energy = self.local_energy.take();
        self.clear();
        let evidence = if let Some((local, prescribed, normalization)) = local_energy {
            let certificate = CoercivityCertificate::prove_local_energy(
                matrix,
                &prescribed,
                local,
                normalization,
            )?;
            PreparedSolverAdmissibility::bind_existing_with_source(
                certificate,
                matrix,
                rhs,
                source,
            )?
        } else {
            PreparedSolverAdmissibility::prove(matrix, rhs, source)?
        };
        self.certificate = Some(evidence.coercivity().clone());
        Ok(evidence)
    }
}

#[cfg(test)]
mod tests;
