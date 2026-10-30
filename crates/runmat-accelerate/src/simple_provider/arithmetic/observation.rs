//! Witnesses are issued by completed native hooks, never by capability lookup.
use super::{InProcessProvider, Operation};
use runmat_accelerate_api::ProviderOperationIdentity;

impl Operation {
    pub(in crate::simple_provider) fn for_identity(
        identity: &ProviderOperationIdentity,
    ) -> Option<Self> {
        use runmat_builtins::{
            BinaryArithmeticInferenceRule as Binary, BuiltinInferenceRule, MathInferenceRule,
        };
        let rule = runmat_builtins::builtin_catalog_entry_by_name(&identity.0)?
            .contract
            .inference_rule;
        match rule {
            BuiltinInferenceRule::Math(MathInferenceRule::BinaryArithmetic(operation)) => {
                match operation {
                    Binary::Add => Some(Self::Add),
                    Binary::Subtract => Some(Self::Sub),
                    Binary::Multiply => Some(Self::Mul),
                    Binary::RightDivide | Binary::LeftDivide => Some(Self::Div),
                    Binary::Power => None,
                }
            }
            _ => None,
        }
    }
}

impl InProcessProvider {
    pub(super) fn confirm_native_arithmetic(&self, completed: Operation) {
        let Some(recorder) = runmat_accelerate_api::observation::current_provider_observation()
        else {
            return;
        };
        let Some(snapshot) = self.capability_snapshot.get() else {
            return;
        };
        // Reuse the provider-owned identity: confirmation allocates no new key.
        if let Some(operation) = snapshot.operations.iter().find(|operation| {
            Operation::for_identity(&operation.identity) == Some(completed)
                && recorder.expects_operation(&operation.identity)
        }) {
            recorder.confirm_provider_execution(self.device_id, &operation.identity);
        }
    }
}
