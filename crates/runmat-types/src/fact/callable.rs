use super::ValueFact;
use crate::{CallContract, CallableIdentity, CapabilitySet, DynamicReason, EffectKind, EffectSet};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallableFact {
    pub identity: Option<CallableIdentity>,
    /// Semantic host capabilities required when this callable executes.
    /// This is part of the value fact so requirements survive assignment,
    /// capture, branch joins, and indirect calls.
    #[serde(default)]
    pub capabilities: CapabilitySet,
    pub parameters: Vec<ValueFact>,
    pub parameters_complete: bool,
    pub outputs: Vec<ValueFact>,
    pub outputs_complete: bool,
    pub variadic_inputs: bool,
    pub variadic_outputs: bool,
    pub captures: Vec<ValueFact>,
    pub captures_complete: bool,
}

impl CallableFact {
    /// Build the call contract carried by this callable value.
    ///
    /// Identity-aware catalogs may refine this contract before inference. This
    /// representation remains authoritative for anonymous, bound, imported,
    /// and otherwise unresolved callables whose output facts travel with the
    /// value itself.
    pub fn call_contract(&self, unresolved: DynamicReason) -> CallContract {
        CallContract {
            outputs: self.outputs.clone(),
            variadic_output: (self.variadic_outputs || !self.outputs_complete)
                .then(|| Box::new(ValueFact::unknown(unresolved.clone()))),
            maximum_outputs: (self.outputs_complete && !self.variadic_outputs)
                .then_some(self.outputs.len()),
            effects: if self.identity.is_none() {
                EffectSet(
                    [
                        EffectKind::HostCallback,
                        EffectKind::MaySuspend,
                        EffectKind::MayThrow,
                        EffectKind::Unknown,
                    ]
                    .into_iter()
                    .collect(),
                )
            } else {
                Default::default()
            },
            capabilities: self.capabilities.clone(),
            dynamic_reason: (!self.outputs_complete || self.variadic_outputs).then_some(unresolved),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unresolved_callable_contract_preserves_results_without_claiming_purity() {
        let output = ValueFact::unknown(DynamicReason::RuntimeValue);
        let mut callable = CallableFact {
            identity: None,
            capabilities: CapabilitySet::default(),
            parameters: Vec::new(),
            parameters_complete: false,
            outputs: vec![output.clone()],
            outputs_complete: true,
            variadic_inputs: true,
            variadic_outputs: false,
            captures: Vec::new(),
            captures_complete: false,
        };
        let contract = callable.call_contract(DynamicReason::UnresolvedCallable);
        assert_eq!(contract.outputs, vec![output]);
        assert_eq!(contract.maximum_outputs, Some(1));
        assert!(contract.variadic_output.is_none());
        assert!(!contract.effects.is_side_effect_free());
        for effect in [
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
            EffectKind::Unknown,
        ] {
            assert!(contract.effects.0.contains(&effect));
        }
        callable.identity = Some(CallableIdentity::Builtin(crate::BuiltinId("sin".into())));
        assert!(callable
            .call_contract(DynamicReason::UnresolvedCallable)
            .effects
            .0
            .is_empty());
    }
}
