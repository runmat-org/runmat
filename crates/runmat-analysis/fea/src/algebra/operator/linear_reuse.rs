//! One bounded process-local proof, rebound only to its exact constrained operator.
use super::*;
use crate::algebra::CoercivityCertificate;

#[derive(Default)]
pub(crate) struct LinearPreparationReuse {
    certificate: Option<CoercivityCertificate>,
    /// Factorization behind a sparse Cholesky certificate; reused for every solve with the same
    /// operator (time steps, eigen iterations, load cases).
    factor: Option<crate::algebra::coercivity::SharedFactor>,
    local_energy: Option<LocalEnergySource>,
}

enum LocalEnergySource {
    Identity(
        crate::algebra::anchored_energy::AnchoredEnergySource,
        Vec<bool>,
        f64,
    ),
    Selection(
        crate::algebra::anchored_energy::AnchoredEnergySource,
        Vec<Option<usize>>,
    ),
}

impl LinearPreparationReuse {
    pub(crate) const RESIDENT_BYTES: u64 = std::mem::size_of::<Self>() as u64;

    pub(crate) fn clear(&mut self) {
        self.certificate = None;
        self.factor = None;
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
            factor: None,
            local_energy: Some(LocalEnergySource::Identity(
                source,
                prescribed,
                normalization,
            )),
        }
    }

    pub(crate) fn with_selected_energy(
        source: crate::algebra::anchored_energy::AnchoredEnergySource,
        rows: Vec<Option<usize>>,
    ) -> Self {
        Self {
            certificate: None,
            factor: None,
            local_energy: Some(LocalEnergySource::Selection(source, rows)),
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
                self.factor.clone(),
                matrix,
                rhs,
                source,
            );
        }
        // A failed replacement cannot leave a stale proof available for later work.
        let local_energy = self.local_energy.take();
        self.clear();
        let evidence = if let Some(local) = local_energy {
            let certificate = match local {
                LocalEnergySource::Identity(local, prescribed, normalization) => {
                    CoercivityCertificate::prove_local_energy(
                        matrix,
                        &prescribed,
                        local,
                        normalization,
                    )
                }
                LocalEnergySource::Selection(local, rows) => {
                    CoercivityCertificate::prove_selected_energy(matrix, local, &rows)
                }
            };
            match certificate {
                Ok(certificate) => PreparedSolverAdmissibility::bind_existing_with_source(
                    certificate,
                    None,
                    matrix,
                    rhs,
                    source,
                )?,
                // A local-energy bound too weak to separate the assembly error from zero is not
                // a disproof: the sparse Cholesky proof decides, and its factor serves the solve.
                Err(CoercivityError::EnergyIndeterminate { .. }) => {
                    PreparedSolverAdmissibility::prove(matrix, rhs, source)?
                }
                Err(error) => return Err(error),
            }
        } else {
            PreparedSolverAdmissibility::prove(matrix, rhs, source)?
        };
        self.certificate = Some(evidence.coercivity().clone());
        self.factor = evidence.factor().cloned();
        Ok(evidence)
    }
}

#[cfg(test)]
mod tests;
