use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinDocumentation, BuiltinErrorDescriptor, BuiltinFusionPolicy,
    BuiltinInferenceRule, BuiltinIntegerCapabilityDescriptor, BuiltinLinkContract,
    BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinPlacementContract, BuiltinPortability, BuiltinPurity,
    BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind, BuiltinSignatureDescriptor,
    MathInferenceRule, NumericLimitRule,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub(super) const CLASS_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "typename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Supported numeric class name.",
}];
pub(super) const LIKE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Like name-value keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Prototype whose class, complexity, sparsity, and applicable residency are copied.",
    },
];
pub(super) const VALUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar numeric limit in the selected representation.",
}];

pub const NUMERIC_LIMIT_ERROR_INVALID_CLASS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUMERIC_LIMITS.INVALID_CLASS",
    identifier: Some("RunMat:numericLimits:InvalidClass"),
    when: "The requested class or prototype representation is not supported by the limit query.",
    message: "numeric limit: unsupported class",
};
pub const NUMERIC_LIMIT_ERROR_INVALID_SYNTAX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUMERIC_LIMITS.INVALID_SYNTAX",
    identifier: Some("RunMat:numericLimits:InvalidSyntax"),
    when: "The arguments do not match a documented type-name or like-prototype form.",
    message: "numeric limit: invalid syntax",
};
pub const NUMERIC_LIMIT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUMERIC_LIMITS.INTERNAL",
    identifier: Some("RunMat:numericLimits:Internal"),
    when: "Constructing or placing the scalar result fails internally.",
    message: "numeric limit: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    NUMERIC_LIMIT_ERROR_INVALID_CLASS,
    NUMERIC_LIMIT_ERROR_INVALID_SYNTAX,
    NUMERIC_LIMIT_ERROR_INTERNAL,
];

pub(super) const fn descriptor(
    signatures: &'static [BuiltinSignatureDescriptor],
) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    }
}

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];

pub(super) const fn entry(
    name: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    rule: NumericLimitRule,
    integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        identity: BuiltinCatalogIdentity { name },
        category: "math/elementwise",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Complete,
            inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::NumericLimit(rule)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &EFFECTS,
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Optional,
            residency: BuiltinResidencyPolicy::Dynamic,
            fusion: BuiltinFusionPolicy::Never,
            distributed: BuiltinDistributedPolicy::ScalarLikePrototype,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &BINDINGS,
        extensions: &[],
        integer_capabilities,
        integer_audit: None,
        suppress_auto_output: false,
    }
}
