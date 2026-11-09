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
    /// Elastic rigidity proof from the assembly that built the operator. It is the only proof
    /// tried: a refusal is final, never a fallback to factorization.
    elastic: Option<crate::algebra::elastic_rigidity::ElasticRigidityProof>,
    /// Structural steady-conduction proof from the assembly that built the operator. Like the
    /// elastic proof it is the only proof tried; a refusal never falls back to factorization.
    conduction: Option<crate::algebra::ScalarConductionProof>,
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

    /// The certificate of the last prepared operator, if any.
    pub(crate) fn certificate(&self) -> Option<&CoercivityCertificate> {
        self.certificate.as_ref()
    }

    pub(crate) fn clear(&mut self) {
        self.certificate = None;
        self.factor = None;
        self.local_energy = None;
        self.elastic = None;
        self.conduction = None;
    }

    pub(crate) fn with_scalar_conduction(proof: crate::algebra::ScalarConductionProof) -> Self {
        Self {
            conduction: Some(proof),
            ..Self::default()
        }
    }

    pub(crate) fn with_elastic_rigidity(
        proof: crate::algebra::elastic_rigidity::ElasticRigidityProof,
    ) -> Self {
        Self {
            elastic: Some(proof),
            ..Self::default()
        }
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
            elastic: None,
            conduction: None,
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
            elastic: None,
            conduction: None,
        }
    }

    pub(super) async fn prepare_admitted_with_owner<O, F, A>(
        &mut self,
        matrix: &RealSparseMatrix,
        rhs: &[f64],
        source: Option<&NullspaceEliminationEvidence>,
        admission: Option<&crate::planning::resource::SolverAdmissionResourcePolicy>,
        before_factor: O,
    ) -> Result<
        PreparedSolverAdmissibility,
        crate::algebra::coercivity::NumericalCapacityError<CoercivityError, A>,
    >
    where
        O: FnMut(
            crate::algebra::coercivity::NumericalCapacityPhase,
            runmat_simulation_usage::OperationCount,
        ) -> F,
        F: std::future::Future<Output = Result<(), A>>,
    {
        if let Some(certificate) = self.certificate.as_ref().filter(|certificate| {
            certificate.validate_for(matrix).is_ok()
                && admission.is_none_or(|policy| {
                    policy.admits_certificate(certificate)
                        && self.factor.as_ref().is_none_or(|factor| {
                            factor.0.symbolic().workspace_policy()
                                == runmat_sparse_solvers::direct::DirectWorkspacePolicy::CallerOwned
                        })
                })
        }) {
            return PreparedSolverAdmissibility::bind_existing_with_source(
                certificate.clone(),
                self.factor.clone(),
                matrix,
                rhs,
                source,
            )
            .map_err(crate::algebra::coercivity::NumericalCapacityError::Numerical);
        }
        // A failed replacement cannot leave a stale proof available for later work.
        let local_energy = self.local_energy.take();
        let elastic = self.elastic.take();
        let conduction = self.conduction.take();
        self.clear();
        let evidence = if let Some(proof) = conduction {
            PreparedSolverAdmissibility::bind_existing_with_source(
                CoercivityCertificate::from_scalar_conduction(matrix, &proof)?,
                None,
                matrix,
                rhs,
                source,
            )?
        } else if let Some(proof) = elastic {
            PreparedSolverAdmissibility::bind_existing_with_source(
                CoercivityCertificate::from_elastic_rigidity(matrix, &proof)?,
                None,
                matrix,
                rhs,
                source,
            )?
        } else if let Some(local) = local_energy {
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
                    PreparedSolverAdmissibility::prove_admitted_with_owner(
                        matrix,
                        rhs,
                        source,
                        admission,
                        before_factor,
                    )
                    .await?
                }
                Err(error) => return Err(error.into()),
            }
        } else {
            PreparedSolverAdmissibility::prove_admitted_with_owner(
                matrix,
                rhs,
                source,
                admission,
                before_factor,
            )
            .await?
        };
        self.certificate = Some(evidence.coercivity().clone());
        self.factor = evidence.factor().cloned();
        Ok(evidence)
    }
}

#[cfg(test)]
mod tests;
