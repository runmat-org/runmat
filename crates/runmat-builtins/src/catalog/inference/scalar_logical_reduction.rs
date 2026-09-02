use super::{argument_error, finish_fixed};
use crate::{BuiltinCatalogEntry, ScalarLogicalReduction};
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact, ValueKindFact};

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    _reduction: ScalarLogicalReduction,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SCALAR-LOGICAL-REDUCTION-ARITY",
            format!("{} requires exactly one input", entry.identity.name),
            request.arguments.len().min(1),
        ));
    }
    if request.arguments.is_empty() {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    }
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Logical),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BuiltinAsyncBehavior, BuiltinBindingAvailability, BuiltinBindingDeclaration,
        BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
        BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation, BuiltinFusionPolicy,
        BuiltinInferenceRule, BuiltinLinkContract, BuiltinOutputMode, BuiltinPlacementContract,
        BuiltinPurity, BuiltinSemanticKind, LogicalInferenceRule,
    };
    use runmat_types::{OutputSelection, RequestedOutputCount};

    #[test]
    fn scalar_reduction_returns_one_logical_fact() {
        let result = infer(
            &CallRequest {
                arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                literals: Default::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &test_entry(),
            ScalarLogicalReduction::AllFinite,
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
        assert!(result.outputs[0].is_scalar());
    }

    #[test]
    fn scalar_reduction_reports_invalid_arity() {
        let result = infer(
            &CallRequest {
                arguments: vec![],
                literals: Default::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &test_entry(),
            ScalarLogicalReduction::AllFinite,
        );
        assert!(!result.diagnostics.is_empty());
    }

    fn test_entry() -> BuiltinCatalogEntry {
        static DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &[],
        };
        static BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
            variant: "default",
            availability: BuiltinBindingAvailability::Required,
        }];
        BuiltinCatalogEntry {
            identity: crate::BuiltinCatalogIdentity {
                name: "scalar-reduction-test",
            },
            category: "logical/tests",
            documentation: BuiltinDocumentation {
                summary: "test",
                example_exemption: Some("private inference fixture"),
                ..BuiltinDocumentation::EMPTY
            },
            descriptor: &DESCRIPTOR,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: BuiltinInferenceRule::Logical(
                    LogicalInferenceRule::ScalarReduction(ScalarLogicalReduction::AllFinite),
                ),
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                purity: BuiltinPurity::Pure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &[],
                capabilities: &[],
            },
            placement: BuiltinPlacementContract {
                portability: crate::BuiltinPortability::NativeAndWasm,
                accelerator: crate::BuiltinAcceleratorPolicy::Optional,
                residency: crate::BuiltinResidencyPolicy::Host,
                fusion: BuiltinFusionPolicy::Boundary,
                distributed: crate::BuiltinDistributedPolicy::MaterializeArguments,
            },
            link: BuiltinLinkContract {
                reachability: crate::BuiltinReachability::Always,
                policy: crate::BuiltinLinkPolicy::PortableRuntime,
                execution_stack: runmat_types::ExecutionStackRequirement::Any,
                artifact_dependencies: &[],
            },
            bindings: &BINDINGS,
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        }
    }
}
