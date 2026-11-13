//! Shared deep-equality declaration and inference; composition parents remain semantic-free.
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, ValueFact, ValueKindFact};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = (request.arguments.len() < 2)
        .then(|| {
            argument_error(
                "RM-CATALOG-EQUALITY-ARITY",
                "equality requires at least two inputs",
                0,
            )
        })
        .into_iter()
        .collect();
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Logical),
        diagnostics,
    )
}
macro_rules! define_equality {
    ($entry:ident,
    $descriptor:ident,
    $arity:ident,
    $internal:ident,
    $outputs:ident,
    $name:literal,
    $upper:literal,
    $documentation:expr) => {
        const INPUTS: &[crate::BuiltinParamDescriptor] = &[
            crate::BuiltinParamDescriptor {
                name: "A",
                ty: crate::BuiltinParamType::Any,
                arity: crate::BuiltinParamArity::Required,
                default: None,
                description: "First value."
            },
            crate::BuiltinParamDescriptor {
                name: "B",
                ty: crate::BuiltinParamType::Any,
                arity: crate::BuiltinParamArity::Required,
                default: None,
                description: "Second value."
            },
            crate::BuiltinParamDescriptor {
                name: "others",
                ty: crate::BuiltinParamType::Any,
                arity: crate::BuiltinParamArity::Variadic,
                default: None,
                description: "Additional values compared against A."
            },
        ];
        const OUTPUTS: &[crate::BuiltinParamDescriptor] = &[
            crate::BuiltinParamDescriptor {
                name: "tf",
                ty: crate::BuiltinParamType::LogicalArray,
                arity: crate::BuiltinParamArity::Required,
                default: None,
                description: "Host logical scalar; numeric values compare across classes without lossy conversion."
            }
        ];
        const SIGNATURES: &[crate::BuiltinSignatureDescriptor] = &[
            crate::BuiltinSignatureDescriptor {
                label: concat!("tf = ",
                $name,
                "(A, B, ...)"),
                inputs: INPUTS,
                outputs: OUTPUTS
            }
        ];
        pub const $arity: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.",
            $upper,
            ".NOT_ENOUGH_INPUTS"),
            identifier: Some(concat!("RunMat:",
            $name,
            ":NotEnoughInputs")),
            when: "Fewer than two inputs are supplied.",
            message: concat!($name,
            ": requires at least two input arguments"),
        };
        pub const $internal: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.",
            $upper,
            ".INTERNAL"),
            identifier: Some(concat!("RunMat:",
            $name,
            ":Internal")),
            when: "Resident input ownership, payload validation, or download fails.",
            message: concat!($name,
            ": internal error"),
        };
        pub const $outputs: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.",
            $upper,
            ".TOO_MANY_OUTPUTS"),
            identifier: Some(concat!("RunMat:",
            $name,
            ":TooManyOutputs")),
            when: "More than one output is requested.",
            message: concat!($name,
            ": too many output arguments"),
        };
        pub const $descriptor: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
            signatures: SIGNATURES,
            output_mode: crate::BuiltinOutputMode::Fixed,
            completion_policy: crate::BuiltinCompletionPolicy::Public,
            errors: &[
                $arity,
                $internal,
                $outputs
            ],
        };
        const INTEGER_INPUTS: &[crate::BuiltinIntegerInputCapability] = &[
            crate::BuiltinIntegerInputCapability {
                name: "A, B, others",
                classes: &crate::ALL_INTEGER_CLASSES,
                availability: crate::BuiltinIntegerInputAvailability::Documented,
                scalar_double: crate::BuiltinIntegerScalarDoubleRule::Allowed,
                notes: "Compare integer, floating, logical and character numeric values exactly across classes; no preconversion of wide integers.",
            }
        ];
        const INTEGER: &[crate::BuiltinIntegerCapabilityDescriptor] = &[
            crate::BuiltinIntegerCapabilityDescriptor {
                form: concat!("tf = ",
                $name,
                "(integer_A, B, ...)"),
                inputs: INTEGER_INPUTS,
                computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
                output_class: crate::BuiltinIntegerOutputClassRule::Logical,
                overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
                backend: crate::BuiltinIntegerBackendRule::GatherFallback,
                overload: crate::BuiltinIntegerOverloadKind::Multiple,
                notes: "Exact class-insensitive comparison, including sparse and complex integer storage and nested containers. Result is one host logical.",
            }
        ];
        pub const $entry: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: crate::BuiltinCatalogProvenance::new(file!(),
            module_path!()),
            identity: crate::BuiltinCatalogIdentity {
                name: $name
            },
            category: "logical/rel",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: crate::BuiltinContractDeclaration {
                maturity: crate::BuiltinContractMaturity::Complete,
                inference_rule: crate::BuiltinInferenceRule::Identity(crate::IdentityInferenceRule::new(super::equality_contract::infer)),
                compatibility: crate::BuiltinCompatibility::Matlab,
                async_behavior: crate::BuiltinAsyncBehavior::MaySuspend,
                purity: crate::BuiltinPurity::Pure,
                semantic_kind: crate::BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &[
                    runmat_types::EffectKind::MaySuspend,
                    runmat_types::EffectKind::MayThrow
                ],
                capabilities: &[
                ],
            },
            placement: crate::BuiltinPlacementContract {
                preparation: crate::BuiltinOperandPreparation::RuntimeOwned,
                portability: crate::BuiltinPortability::NativeAndWasm,
                accelerator: crate::BuiltinAcceleratorPolicy::Optional,
                residency: crate::BuiltinResidencyPolicy::GatherToHost,
                fusion: crate::BuiltinFusionPolicy::Boundary,
                distributed: crate::BuiltinDistributedPolicy::MaterializeArguments,
            },
            link: crate::BuiltinLinkContract {
                reachability: crate::BuiltinReachability::Always,
                policy: crate::BuiltinLinkPolicy::PortableRuntime,
                execution_stack: runmat_types::ExecutionStackRequirement::Any,
                artifact_dependencies: &[
                ],
            },
            bindings: &crate::REQUIRED_DEFAULT_BINDING,
            extensions: &[
            ],
            integer_capabilities: INTEGER,
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}
pub(super) use define_equality;
#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{DynamicReason, LiteralContext, OutputSelection, RequestedOutputCount};
    #[test]
    fn output_is_logical_scalar_for_known_and_unknown_inputs() {
        for entry in [
            &super::super::isequal::ISEQUAL_CATALOG_ENTRY,
            &super::super::isequaln::ISEQUALN_CATALOG_ENTRY,
        ] {
            for argument in [
                ValueFact::scalar(ValueKindFact::Logical),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ] {
                let mut request = CallRequest {
                    arguments: vec![argument.clone(), argument],
                    literals: LiteralContext::default(),
                    outputs: OutputSelection::new(RequestedOutputCount::One),
                };
                let result = infer(&request, entry);
                assert_eq!(
                    result.outputs,
                    vec![ValueFact::scalar(ValueKindFact::Logical)]
                );
                assert!(result.diagnostics.is_empty());
                request.outputs = OutputSelection::new(RequestedOutputCount::Zero);
                assert!(infer(&request, entry).outputs.is_empty());
                request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
                assert!(!infer(&request, entry).diagnostics.is_empty());
                request.arguments.clear();
                request.outputs = OutputSelection::new(RequestedOutputCount::One);
                assert!(!infer(&request, entry).diagnostics.is_empty());
            }
        }
    }
}
