//! Closed typed facts carrier. Operation domains own derivation and checks;
//! shared placement only verifies the carrier matches its typed workload.

use super::*;
use runmat_builtins::{AudioDecodeResourceFacts, ReadMatrixTextResourceFacts};

#[derive(Clone, Copy)]
pub enum HostEffectFinalFacts<'a> {
    AudioDecode(&'a AudioDecodeResourceFacts),
    ReadMatrixText(&'a ReadMatrixTextResourceFacts),
    #[cfg(test)]
    TestOnly(Digest),
}

impl HostEffectFinalFacts<'_> {
    pub(super) fn validate_for(self, request: &HostEffectRequest<'_>) -> BuiltinResult<()> {
        let compatible = match (self, request.workload) {
            (Self::AudioDecode(_), HostEffectWorkload::AudioDecode { .. }) => true,
            (Self::ReadMatrixText(_), HostEffectWorkload::ReadMatrixText { .. }) => true,
            #[cfg(test)]
            (Self::TestOnly(_), _) => true,
            _ => false,
        };
        if !compatible || request.domain_facts != self.digest() {
            return Err(stage::invalid_stage(
                "final facts do not match the staged workload",
            ));
        }
        Ok(())
    }

    fn digest(self) -> Digest {
        match self {
            Self::AudioDecode(facts) => {
                Digest::sha256(serde_json::to_vec(facts).expect("typed WAVE facts serialize"))
            }
            Self::ReadMatrixText(facts) => super::hash_serializable(facts),
            #[cfg(test)]
            Self::TestOnly(digest) => digest,
        }
    }
}
