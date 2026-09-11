use super::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact, ValueKindFact};

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    diagnostic_code: &'static str,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            diagnostic_code,
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
        BuiltinPurity, BuiltinSemanticKind, LogicalInferenceRule, MetadataPredicate,
    };
    use runmat_types::{OutputSelection, RequestedOutputCount};

    #[test]
    fn unary_rule_returns_one_logical_scalar() {
        let result = infer(
            &CallRequest {
                arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                literals: Default::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &test_entry(),
            "RM-CATALOG-TEST-ARITY",
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
        assert!(result.outputs[0].is_scalar());
    }

    #[test]
    fn unary_rule_reports_invalid_arity() {
        let result = infer(
            &CallRequest {
                arguments: vec![],
                literals: Default::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &test_entry(),
            "RM-CATALOG-TEST-ARITY",
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
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: crate::BuiltinCatalogIdentity { name: "unary-test" },
            category: "test",
            documentation: BuiltinDocumentation {
                summary: "test",
                example_exemption: Some("private inference fixture"),
                ..BuiltinDocumentation::EMPTY
            },
            descriptor: &DESCRIPTOR,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: BuiltinInferenceRule::Logical(
                    LogicalInferenceRule::MetadataPredicate(MetadataPredicate::Logical),
                ),
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::NeverSuspends,
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
                distributed: crate::BuiltinDistributedPolicy::InspectHandles,
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
