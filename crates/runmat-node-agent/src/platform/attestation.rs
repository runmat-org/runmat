use async_trait::async_trait;

use crate::AgentResult;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationEvidence {
    pub class: String,
    pub evidence: Vec<u8>,
}

#[async_trait]
pub trait AttestationProvider: Send + Sync {
    async fn evidence(&self) -> AgentResult<Option<AttestationEvidence>>;
}

#[derive(Debug, Default)]
pub struct NoAttestation;

#[async_trait]
impl AttestationProvider for NoAttestation {
    async fn evidence(&self) -> AgentResult<Option<AttestationEvidence>> {
        Ok(None)
    }
}
