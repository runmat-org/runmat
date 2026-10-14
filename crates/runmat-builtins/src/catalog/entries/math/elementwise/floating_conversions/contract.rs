pub(super) use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, ALL_INTEGER_CLASSES,
};
pub(super) use runmat_types::{EffectKind, ExecutionStackRequirement, NumericClass};

#[macro_use]
mod descriptors;

macro_rules! define_floating_conversion_contract {
    (
        $name:literal,
        $upper:literal,
        $class:expr,
        $preparation_requirement:expr,
        $preparation_forms:expr,
        $documentation:expr,
        $output_description:literal,
        $gpu_storage_description:literal,
        $extension_id:literal,
        $extension_description:literal,
        $extension_error:literal,
        $integer_backend:expr,
        $integer_output:expr,
        $integer_notes:literal,
        $backend_notes:literal
    ) => {
        const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
            name: "Y",
            ty: BuiltinParamType::NumericArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: $output_description,
        }];
        const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
            name: "X",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Input scalar or array value to convert.",
        }];
        const LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
            BuiltinParamDescriptor {
                name: "X",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Input scalar or array value to convert.",
            },
            BuiltinParamDescriptor {
                name: "like",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Literal string \"like\".",
            },
            BuiltinParamDescriptor {
                name: "prototype",
                ty: BuiltinParamType::LikePrototype,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Output residency prototype; the conversion class remains fixed.",
            },
        ];
        const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
            BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X)"),
                inputs: &INPUT,
                outputs: &OUTPUTS,
            },
            BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X, \"like\", prototype)"),
                inputs: &LIKE_INPUTS,
                outputs: &OUTPUTS,
            },
        ];

        define_floating_conversion_descriptors!(
            $name,
            $upper,
            $gpu_storage_description,
            $extension_id,
            $extension_description,
            $extension_error
        );

        const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
            [BuiltinIntegerInputCapability {
                name: "X",
                classes: &ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::Documented,
                scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: $integer_notes,
            }];
        pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
            [BuiltinIntegerCapabilityDescriptor {
                form: concat!("Y = ", $name, "(integer_X)"),
                inputs: &INTEGER_INPUTS,
                computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
                output_class: $integer_output,
                overflow: BuiltinIntegerOverflowRule::NotApplicable,
                backend: $integer_backend,
                overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
                notes: $backend_notes,
            }];

        const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
        const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];

        // Form 0 retains its original meaning and signature coverage. New
        // audited routes use fresh IDs rather than reinterpret serialized facts.
        const DEFAULT_PREPARATION_FORM: crate::BuiltinPreparationForm =
            crate::BuiltinPreparationForm {
                id: crate::BuiltinPreparationFormId(0),
                binding_variant: "default",
                signatures: &[0, 1],
                data_arguments: &[0],
                operation: crate::BuiltinPreparationOperation::Unary,
                provider_requirement: $preparation_requirement,
            };

        // The like prototype selects output residency inside this domain;
        // single/double determine the conversion class. Preserve legacy form 0
        // and plain-cast form 1 identities;
        // this distinct form does not claim a shared unary physical route.
        const LIKE_PREPARATION_FORM: crate::BuiltinPreparationForm =
            crate::BuiltinPreparationForm {
                id: crate::BuiltinPreparationFormId(2),
                binding_variant: "default",
                signatures: &[1],
                data_arguments: &[0],
                operation: crate::BuiltinPreparationOperation::RuntimeOwned,
                provider_requirement: crate::BuiltinPreparationProviderRequirement::InputSupported,
            };

        pub const CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "math/elementwise",
            documentation: $documentation,
            descriptor: &DESCRIPTOR,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: BuiltinInferenceRule::Math(
                    MathInferenceRule::NumericConversionWithLike($class),
                ),
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                purity: BuiltinPurity::Pure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &EFFECTS,
                capabilities: &[],
            },
            placement: BuiltinPlacementContract {
                preparation: crate::BuiltinOperandPreparation::Forms($preparation_forms),
                portability: BuiltinPortability::NativeAndWasm,
                accelerator: BuiltinAcceleratorPolicy::Optional,
                residency: BuiltinResidencyPolicy::Dynamic,
                fusion: BuiltinFusionPolicy::Candidate,
                distributed: crate::BuiltinDistributedPolicy::MapUnary,
            },
            link: BuiltinLinkContract {
                reachability: BuiltinReachability::Always,
                policy: BuiltinLinkPolicy::PortableRuntime,
                execution_stack: ExecutionStackRequirement::Any,
                artifact_dependencies: &[],
            },
            bindings: &BINDINGS,
            extensions: &EXTENSIONS,
            integer_capabilities: &INTEGER_CAPABILITIES,
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}
