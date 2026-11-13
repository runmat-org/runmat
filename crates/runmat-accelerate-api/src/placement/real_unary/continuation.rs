//! Actual consumed-recipe outcomes, never inferred from output residency.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderRealUnaryContinuationKind {
    Outside,
    Indeterminate,
    RealHookUnsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum ProviderRealUnaryPublication {
    Host,
    ProviderUpload,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct ProviderRealUnaryContinuations {
    pub real: ProviderRealUnaryPublication,
    pub complex: ProviderRealUnaryPublication,
}

pub struct ProviderRealUnaryContinuation {
    evidence: ProviderRealUnaryDomain,
    kind: ProviderRealUnaryContinuationKind,
}
impl ProviderRealUnaryContinuation {
    pub fn kind(&self) -> ProviderRealUnaryContinuationKind {
        self.kind
    }
    pub fn classification(&self) -> RealUnaryClassification {
        self.evidence.classification
    }
    pub(crate) fn recipe(&self) -> Digest {
        self.evidence.recipe
    }
    pub(crate) fn source(&self) -> crate::allocation::TensorViewIdentity {
        self.evidence.source
    }
}

pub struct ProviderRealUnaryExecuted {
    output: GpuTensorHandle,
    evidence: ProviderRealUnaryDomain,
}
pub struct ProviderRealUnaryExecutionEvidence {
    recipe: Digest,
    source: crate::allocation::TensorViewIdentity,
}
impl ProviderRealUnaryExecuted {
    pub fn into_parts(self) -> (GpuTensorHandle, ProviderRealUnaryExecutionEvidence) {
        (
            self.output,
            ProviderRealUnaryExecutionEvidence {
                recipe: self.evidence.recipe,
                source: self.evidence.source,
            },
        )
    }
}
impl ProviderRealUnaryExecutionEvidence {
    pub(crate) fn recipe(&self) -> Digest {
        self.recipe
    }
    pub(crate) fn source(&self) -> crate::allocation::TensorViewIdentity {
        self.source
    }
}

pub enum ProviderRealUnaryExecution {
    Real(ProviderRealUnaryExecuted),
    Continue(ProviderRealUnaryContinuation),
}

impl ProviderRealUnaryCompound {
    pub fn authenticates_continuation(&self, evidence: &ProviderRealUnaryContinuation) -> bool {
        self.authenticates(&evidence.evidence)
    }
    /// Outside is not Unsupported. Only actual consuming execution can mint
    /// the separate pre-effect-real-hook-Unsupported continuation.
    pub async fn consume(
        &self,
        evidence: ProviderRealUnaryDomain,
        cancellation: Option<&CancellationToken>,
    ) -> ProviderExecutionResult<ProviderRealUnaryExecution> {
        check_cancelled(cancellation, true)?;
        if !self.authenticates(&evidence) {
            return Err(unproven().after_effect());
        }
        let kind = match evidence.classification {
            RealUnaryClassification::Outside => ProviderRealUnaryContinuationKind::Outside,
            RealUnaryClassification::Indeterminate => {
                ProviderRealUnaryContinuationKind::Indeterminate
            }
            RealUnaryClassification::Inside => {
                match self.execute_real_child(&evidence, cancellation).await {
                    Ok(executed) => {
                        return Ok(ProviderRealUnaryExecution::Real(
                            ProviderRealUnaryExecuted {
                                output: executed.output,
                                evidence,
                            },
                        ))
                    }
                    Err(cause) if cause.is_unsupported() => {
                        ProviderRealUnaryContinuationKind::RealHookUnsupported
                    }
                    Err(cause) => return Err(cause.after_effect()),
                }
            }
        };
        Ok(ProviderRealUnaryExecution::Continue(
            ProviderRealUnaryContinuation { evidence, kind },
        ))
    }
}
