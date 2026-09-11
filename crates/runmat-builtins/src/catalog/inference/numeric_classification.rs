use super::{argument_error, finish_fixed};
use crate::{BuiltinCatalogEntry, NumericClassificationPredicate};
use runmat_types::{
    AliasFact, CallInference, CallRequest, ContiguityFact, DynamicReason, LayoutFact, MutationFact,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    _predicate: NumericClassificationPredicate,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-CLASSIFICATION-ARITY",
            format!("{} requires exactly one input", entry.identity.name),
            request.arguments.len().min(1),
        ));
    }
    let Some(input) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-CLASSIFICATION-SPARSE",
            format!("{} does not accept sparse input", entry.identity.name),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }
    match input.kind {
        ValueKindFact::Numeric(_)
        | ValueKindFact::Logical
        | ValueKindFact::Character
        | ValueKindFact::String => {}
        ValueKindFact::Unknown => {
            return finish_fixed(
                entry,
                request,
                ValueFact::unknown(DynamicReason::RuntimeValue),
                diagnostics,
            );
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-CLASSIFICATION-INPUT",
                format!(
                    "{} requires numeric, logical, character, or string input",
                    entry.identity.name
                ),
                0,
            ));
            return finish_fixed(
                entry,
                request,
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
                diagnostics,
            );
        }
    }

    let mut output = input.clone();
    output.kind = ValueKindFact::Logical;
    output.storage = if input.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.view = ViewFact::Materialized;
    output.residency = match &input.residency {
        ResidencyFact::Host => ResidencyFact::Host,
        ResidencyFact::Device { .. } => input.residency.clone(),
        _ => ResidencyFact::Unknown,
    };
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BuiltinAsyncBehavior, BuiltinBindingAvailability, BuiltinBindingDeclaration,
        BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
        BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation, BuiltinFusionPolicy,
        BuiltinInferenceRule, BuiltinLinkContract, BuiltinOutputMode, BuiltinPlacementContract,
        BuiltinPurity, BuiltinSemanticKind, LogicalInferenceRule, NumericClassificationPredicate,
    };
    use runmat_types::{
        NumericClass, NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact,
    };

    // Full entry validation belongs to the identity packages. This fixture isolates
    // the reusable fact rule without introducing a name-selected production path.
    fn request(input: ValueFact) -> CallRequest {
        CallRequest {
            arguments: vec![input],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn classification_preserves_shape_and_device_residency() {
        let mut input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        input.residency = ResidencyFact::Device {
            provider: Some("wgpu".into()),
        };
        let entry = test_entry();
        let result = infer(&request(input), &entry, NumericClassificationPredicate::Nan);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert!(matches!(
            result.outputs[0].residency,
            ResidencyFact::Device { .. }
        ));
    }

    #[test]
    fn classification_rejects_sparse_and_container_inputs() {
        let entry = test_entry();
        let mut sparse = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        sparse.storage = StorageFact::Sparse;
        assert!(!infer(
            &request(sparse),
            &entry,
            NumericClassificationPredicate::Finite,
        )
        .diagnostics
        .is_empty());
        assert!(!infer(
            &request(ValueFact::scalar(ValueKindFact::Void)),
            &entry,
            NumericClassificationPredicate::Infinite,
        )
        .diagnostics
        .is_empty());
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
        static DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
            summary: "test",
            example_exemption: Some("private inference fixture"),
            ..BuiltinDocumentation::EMPTY
        };
        BuiltinCatalogEntry {
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: crate::BuiltinCatalogIdentity {
                name: "classification-test",
            },
            category: "logical/tests",
            documentation: DOCUMENTATION,
            descriptor: &DESCRIPTOR,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: BuiltinInferenceRule::Logical(
                    LogicalInferenceRule::NumericClassification(
                        NumericClassificationPredicate::Nan,
                    ),
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
                residency: crate::BuiltinResidencyPolicy::PreserveInputs,
                fusion: BuiltinFusionPolicy::Candidate,
                distributed: crate::BuiltinDistributedPolicy::MapUnary,
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
