//! Complete paired provider alternatives, with actual consumed-probe evidence.
use super::{PowerEvaluationRecipe, PowerProbeRecipe};
use crate::placement::compound_execution::{
    check_cancelled, retire_output_after_failure, with_prior_effect,
};
use crate::{AccelProvider, GpuTensorHandle, PowerResultDomain, ProviderExecutionResult};
use runmat_execution::{cancellation::CancellationToken, Digest};

#[derive(Clone)]
pub struct PowerPairCompound {
    probe: PowerProbeRecipe,
    real: PowerEvaluationRecipe,
    complex: PowerEvaluationRecipe,
}

impl std::fmt::Debug for PowerPairCompound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PowerPairCompound")
            .field(
                "sources",
                &self
                    .probe
                    .sources()
                    .each_ref()
                    .map(GpuTensorHandle::view_identity),
            )
            .finish_non_exhaustive()
    }
}

pub struct PowerPairExecuted {
    output: GpuTensorHandle,
    evidence: PowerPairExecutionEvidence,
}

pub struct PowerPairExecutionEvidence {
    recipe: Digest,
    sources: [crate::allocation::TensorViewIdentity; 2],
    domain: PowerResultDomain,
}

impl PowerPairExecuted {
    pub fn into_parts(self) -> (GpuTensorHandle, PowerPairExecutionEvidence) {
        (self.output, self.evidence)
    }
}
impl PowerPairExecutionEvidence {
    pub fn domain(&self) -> PowerResultDomain {
        self.domain
    }
    pub(crate) fn recipe(&self) -> Digest {
        self.recipe
    }
    pub(crate) fn sources(&self) -> [crate::allocation::TensorViewIdentity; 2] {
        self.sources
    }
}

impl PowerPairCompound {
    pub fn plan(
        owner: &'static dyn AccelProvider,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> ProviderExecutionResult<Option<Self>> {
        let Some(probe) = PowerProbeRecipe::plan(owner, base, exponent)? else {
            return Ok(None);
        };
        Self::from_probe(owner, probe)
    }

    pub(super) fn from_probe(
        owner: &'static dyn AccelProvider,
        probe: PowerProbeRecipe,
    ) -> ProviderExecutionResult<Option<Self>> {
        let Some(real) = PowerEvaluationRecipe::plan(owner, &probe, PowerResultDomain::Real)?
        else {
            return Ok(None);
        };
        let Some(complex) = PowerEvaluationRecipe::plan(owner, &probe, PowerResultDomain::Complex)?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            probe,
            real,
            complex,
        }))
    }
    pub fn probe(&self) -> &PowerProbeRecipe {
        &self.probe
    }
    pub fn real(&self) -> &PowerEvaluationRecipe {
        &self.real
    }
    pub fn complex(&self) -> &PowerEvaluationRecipe {
        &self.complex
    }
    pub fn is_owned_by(&self, owner: &dyn AccelProvider) -> bool {
        self.probe.is_owned_by(owner)
    }
    pub fn identity(&self) -> anyhow::Result<Digest> {
        Ok(Digest::sha256_serialized(&(
            "power-pair-compound-v1",
            self.probe.identity()?,
            self.real.identity()?,
            self.complex.identity()?,
        ))?)
    }

    pub async fn execute(
        &self,
        cancellation: Option<&CancellationToken>,
    ) -> ProviderExecutionResult<PowerPairExecuted> {
        self.execute_admitted(cancellation, None).await
    }

    pub async fn execute_admitted(
        &self,
        cancellation: Option<&CancellationToken>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<PowerPairExecuted> {
        check_cancelled(cancellation, false)?;
        let evidence = self.probe.execute().await?;
        check_cancelled(cancellation, true)?;
        let evaluation = match evidence.domain() {
            PowerResultDomain::Real => &self.real,
            PowerResultDomain::Complex => &self.complex,
        };
        let executed = with_prior_effect(
            evaluation.execute_admitted(&evidence, admission).await,
            true,
        )?;
        let final_identity = with_prior_effect(self.identity().map_err(Into::into), true);
        let completed = check_cancelled(cancellation, true).and(final_identity);
        let recipe = match completed {
            Ok(recipe) => recipe,
            Err(primary) => {
                return Err(retire_output_after_failure(executed.output, primary));
            }
        };
        Ok(PowerPairExecuted {
            output: executed.output,
            evidence: PowerPairExecutionEvidence {
                recipe,
                sources: self
                    .probe
                    .sources()
                    .each_ref()
                    .map(GpuTensorHandle::view_identity),
                domain: evidence.domain(),
            },
        })
    }
}
