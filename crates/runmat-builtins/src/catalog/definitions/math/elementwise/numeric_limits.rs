use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinDocumentation, BuiltinErrorDescriptor, BuiltinFusionPolicy,
    BuiltinInferenceRule, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinLinkContract,
    BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinPlacementContract, BuiltinPortability, BuiltinPurity,
    BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind, BuiltinSignatureDescriptor,
    FloatingLimitKind, IntegerLimitKind, MathInferenceRule, NumericLimitRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

const CLASS_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "typename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Supported numeric class name.",
}];
const LIKE_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const VALUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar numeric limit in the selected representation.",
}];

macro_rules! signatures {
    ($constant:ident, $name:literal) => {
        const $constant: [BuiltinSignatureDescriptor; 3] = [
            BuiltinSignatureDescriptor {
                label: concat!("value = ", $name, "()"),
                inputs: &[],
                outputs: &VALUE_OUTPUT,
            },
            BuiltinSignatureDescriptor {
                label: concat!("value = ", $name, "(typename)"),
                inputs: &CLASS_INPUT,
                outputs: &VALUE_OUTPUT,
            },
            BuiltinSignatureDescriptor {
                label: concat!("value = ", $name, "(\"like\", prototype)"),
                inputs: &LIKE_INPUTS,
                outputs: &VALUE_OUTPUT,
            },
        ];
    };
}

signatures!(INTMIN_SIGNATURES, "intmin");
signatures!(INTMAX_SIGNATURES, "intmax");
signatures!(REALMIN_SIGNATURES, "realmin");
signatures!(REALMAX_SIGNATURES, "realmax");
signatures!(FLINTMAX_SIGNATURES, "flintmax");

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

macro_rules! descriptor {
    ($constant:ident, $signatures:ident) => {
        pub const $constant: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

descriptor!(INTMIN_DESCRIPTOR, INTMIN_SIGNATURES);
descriptor!(INTMAX_DESCRIPTOR, INTMAX_SIGNATURES);
descriptor!(REALMIN_DESCRIPTOR, REALMIN_SIGNATURES);
descriptor!(REALMAX_DESCRIPTOR, REALMAX_SIGNATURES);
descriptor!(FLINTMAX_DESCRIPTOR, FLINTMAX_SIGNATURES);

const INTEGER_LIMIT_LIKE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "prototype",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The prototype selects one of the eight integer classes and may be real or complex.",
    }];
pub const INTMAX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "value = intmax(\"like\", integer_prototype)",
        inputs: &INTEGER_LIMIT_LIKE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns the exact maximum as a scalar with the prototype class, complexity, and applicable residency.",
    }];
pub const INTMIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "value = intmin(\"like\", integer_prototype)",
        inputs: &INTEGER_LIMIT_LIKE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns the exact minimum as a scalar with the prototype class, complexity, and applicable residency.",
    }];

const INTMIN_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const INTMAX_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const REALMIN_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const REALMAX_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const FLINTMAX_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const MAY_THROW: [EffectKind; 1] = [EffectKind::MayThrow];

const fn entry(
    name: &'static str,
    summary: &'static str,
    keywords: &'static [&'static str],
    descriptor: &'static BuiltinDescriptor,
    rule: NumericLimitRule,
    bindings: &'static [BuiltinBindingDeclaration],
    integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        identity: BuiltinCatalogIdentity { name },
        category: "math/elementwise",
        documentation: BuiltinDocumentation {
            summary,
            keywords,
            related: &["intmin", "intmax", "realmin", "realmax", "flintmax", "eps"],
            introduced: None,
            status: None,
            examples: &[],
            ..BuiltinDocumentation::EMPTY
        },
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
            effects: &MAY_THROW,
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
        bindings,
        extensions: &[],
        integer_capabilities,
        integer_audit: None,
        suppress_auto_output: false,
    }
}

pub const INTMIN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "intmin",
    "Return the smallest value of an integer class.",
    &["intmin", "integer", "limits", "like"],
    &INTMIN_DESCRIPTOR,
    NumericLimitRule::Integer(IntegerLimitKind::Minimum),
    &INTMIN_BINDINGS,
    &INTMIN_INTEGER_CAPABILITIES,
);
pub const INTMAX_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "intmax",
    "Return the largest value of an integer class.",
    &["intmax", "integer", "limits", "like"],
    &INTMAX_DESCRIPTOR,
    NumericLimitRule::Integer(IntegerLimitKind::Maximum),
    &INTMAX_BINDINGS,
    &INTMAX_INTEGER_CAPABILITIES,
);
pub const REALMIN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "realmin",
    "Return the smallest positive normalized floating-point value.",
    &["realmin", "floating point", "limits", "like"],
    &REALMIN_DESCRIPTOR,
    NumericLimitRule::Floating(FloatingLimitKind::SmallestNormal),
    &REALMIN_BINDINGS,
    &[],
);
pub const REALMAX_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "realmax",
    "Return the largest finite floating-point value.",
    &["realmax", "floating point", "limits", "like"],
    &REALMAX_DESCRIPTOR,
    NumericLimitRule::Floating(FloatingLimitKind::LargestFinite),
    &REALMAX_BINDINGS,
    &[],
);
pub const FLINTMAX_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "flintmax",
    "Return the largest consecutive integer in a floating-point class.",
    &["flintmax", "floating point", "integer precision", "like"],
    &FLINTMAX_DESCRIPTOR,
    NumericLimitRule::Floating(FloatingLimitKind::LargestConsecutiveInteger),
    &FLINTMAX_BINDINGS,
    &[],
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &INTMIN_CATALOG_ENTRY,
    &INTMAX_CATALOG_ENTRY,
    &REALMIN_CATALOG_ENTRY,
    &REALMAX_CATALOG_ENTRY,
    &FLINTMAX_CATALOG_ENTRY,
];
