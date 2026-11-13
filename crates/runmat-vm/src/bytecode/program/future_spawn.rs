//! Nonwire compiler evidence retained with its selected immutable body.
#[derive(Clone, Debug)]
pub struct CompilerSpawnContract(runmat_types::SpawnSafetyFact);

impl CompilerSpawnContract {
    pub(crate) fn issue(safety: runmat_types::SpawnSafetyFact) -> Self {
        Self(safety)
    }
    pub fn safety(&self) -> &runmat_types::SpawnSafetyFact {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_body_clone_retains_local_evidence_but_wire_reconstruction_cannot_supply_it() {
        for safety in [
            runmat_types::SpawnSafetyFact::SpawnSafe,
            runmat_types::SpawnSafetyFact::NotSpawnSafe {
                reason: runmat_types::SpawnSafetyReason::MutableLexicalCapture,
            },
        ] {
            let body = crate::bytecode::FunctionBytecode {
                spawn_contract: Some(CompilerSpawnContract::issue(safety.clone())),
                ..Default::default()
            };
            assert_eq!(body.clone().spawn_contract.unwrap().safety(), &safety);
            let mut wire = serde_json::to_value(&body).unwrap();
            assert!(wire.get("spawn_contract").is_none());
            wire["spawn_contract"] = serde_json::json!("SpawnSafe");
            let restored: crate::bytecode::FunctionBytecode = serde_json::from_value(wire).unwrap();
            assert!(restored.spawn_contract.is_none());
            assert_eq!(
                super::super::semantic_definition_fingerprint(&body).unwrap(),
                super::super::semantic_definition_fingerprint(&restored).unwrap(),
            );
        }
    }
}
