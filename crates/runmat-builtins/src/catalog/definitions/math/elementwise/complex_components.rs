use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation,
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy,
    BuiltinInferenceRule, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinLinkContract,
    BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinPlacementContract, BuiltinPortability, BuiltinPurity,
    BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind, BuiltinSignatureDescriptor,
    MathInferenceRule, NumericComponentRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];

macro_rules! define_component_projection {
    (
        $module:ident,
        $name:literal,
        $upper:literal,
        $rule:expr,
        $summary:literal,
        $output_description:literal,
        $integer_notes:literal,
        $capability_notes:literal
    ) => {
        mod $module {
            use super::*;

            const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
                name: "Y",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: $output_description,
            }];
            const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
                name: "X",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Numeric, logical, character, or complex input.",
            }];
            const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X)"),
                inputs: &INPUTS,
                outputs: &OUTPUTS,
            }];

            pub const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
                code: concat!("RM.", $upper, ".INVALID_INPUT"),
                identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
                when:
                    "Input cannot be interpreted as numeric, logical, character, or complex data.",
                message: concat!($name, ": invalid input"),
            };
            pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
                code: concat!("RM.", $upper, ".INTERNAL"),
                identifier: Some(concat!("RunMat:", $name, ":Internal")),
                when: "Internal tensor conversion, allocation, or provider interaction fails.",
                message: concat!($name, ": internal error"),
            };
            const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];
            pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
                signatures: &SIGNATURES,
                output_mode: BuiltinOutputMode::Fixed,
                completion_policy: BuiltinCompletionPolicy::Public,
                errors: &ERRORS,
            };

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
                    computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
                    output_class: BuiltinIntegerOutputClassRule::PreserveInput,
                    overflow: BuiltinIntegerOverflowRule::NotApplicable,
                    backend: BuiltinIntegerBackendRule::HostAndGpu,
                    overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
                    notes: $capability_notes,
                }];

            const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;

            pub const CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
                identity: BuiltinCatalogIdentity { name: $name },
                category: "math/elementwise",
                documentation: BuiltinDocumentation {
                    summary: $summary,
                    keywords: &[$name, "complex", "component", "elementwise", "gpu"],
                    related: &["complex", "conj", "real", "imag", "angle"],
                    introduced: None,
                    status: None,
                    examples: &[],
                },
                descriptor: &DESCRIPTOR,
                contract: BuiltinContractDeclaration {
                    maturity: BuiltinContractMaturity::Complete,
                    inference_rule: BuiltinInferenceRule::Math(
                        MathInferenceRule::NumericComponent($rule),
                    ),
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
                    residency: BuiltinResidencyPolicy::PreserveInputs,
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
                extensions: &[],
                integer_capabilities: &INTEGER_CAPABILITIES,
                integer_audit: None,
                suppress_auto_output: false,
            };
        }
    };
}

define_component_projection!(
    real,
    "real",
    "REAL",
    NumericComponentRule::RealPart,
    "Extract the real component of numeric values.",
    "Real component of X.",
    "All eight integer classes retain their class while projecting the real component.",
    "Real integer input is an exact same-class identity. Paired complex-integer input projects its same-class real storage without arithmetic; supported resident forms preserve class, shape, owner, and residency."
);

define_component_projection!(
    imag,
    "imag",
    "IMAG",
    NumericComponentRule::ImaginaryPart,
    "Extract the imaginary component of numeric values.",
    "Imaginary component of X.",
    "All eight real and componentwise-complex integer classes retain their class.",
    "Real integer input produces exact same-class zeros. Paired complex-integer input projects its exact imaginary storage without arithmetic or overflow; supported resident forms preserve class, shape, owner, and residency."
);

mod conj {
    use super::*;

    const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex conjugate of X.",
    }];
    const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, character, or complex input.",
    }];
    const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
        label: "Y = conj(X)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    }];

    pub const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
        code: "RM.CONJ.INVALID_INPUT",
        identifier: Some("RunMat:conj:InvalidInput"),
        when: "Input cannot be interpreted as numeric, logical, character, or complex data.",
        message: "conj: invalid input",
    };
    pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
        code: "RM.CONJ.INTERNAL",
        identifier: Some("RunMat:conj:Internal"),
        when: "Internal tensor conversion, allocation, or provider interaction fails.",
        message: "conj: internal error",
    };
    const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];
    pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

    pub const CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "conj-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "conj with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ConjCharacterInputExtension"),
    };
    pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [CHARACTER_INPUT_EXTENSION];

    const REAL_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
        [BuiltinIntegerInputCapability {
            name: "X",
            classes: &ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::Documented,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "All eight real integer classes are exact identity inputs.",
        }];
    const COMPLEX_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
        [BuiltinIntegerInputCapability {
            name: "X",
            classes: &ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::Documented,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "Typed complex integer storage retains its class; imaginary-component negation saturates while signed-minimum and unsigned endpoint compatibility remains evidence-open.",
        }];
    pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
        BuiltinIntegerCapabilityDescriptor {
            form: "Y = conj(real_integer_X)",
            inputs: &REAL_INTEGER_INPUT,
            computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
            output_class: BuiltinIntegerOutputClassRule::PreserveInput,
            overflow: BuiltinIntegerOverflowRule::NotApplicable,
            backend: BuiltinIntegerBackendRule::HostAndGpu,
            overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
            notes: "Host inputs are returned unchanged; supported resident real integer inputs preserve the same exact handle and metadata.",
        },
        BuiltinIntegerCapabilityDescriptor {
            form: "Y = conj(complex_integer_X)",
            inputs: &COMPLEX_INTEGER_INPUT,
            computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
            output_class: BuiltinIntegerOutputClassRule::PreserveInput,
            overflow: BuiltinIntegerOverflowRule::EvidenceOpen,
            backend: BuiltinIntegerBackendRule::HostAndGpu,
            overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
            notes: "Typed-complex integer storage uses saturating imaginary negation, preserves paired native storage on the host and owning provider, and retains the evidence qualification for signed-minimum and unsigned-imaginary endpoints.",
        },
    ];

    const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;

    pub const CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
        identity: BuiltinCatalogIdentity { name: "conj" },
        category: "math/elementwise",
        documentation: BuiltinDocumentation {
            summary: "Compute complex conjugates element-wise.",
            keywords: &["conj", "complex conjugate", "complex", "elementwise", "gpu"],
            related: &["complex", "real", "imag", "angle"],
            introduced: None,
            status: None,
            examples: &[],
        },
        descriptor: &DESCRIPTOR,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Complete,
            inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::NumericComponent(
                NumericComponentRule::Conjugate,
            )),
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
            residency: BuiltinResidencyPolicy::PreserveInputs,
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
}

pub use conj::{
    CATALOG_ENTRY as CONJ_CATALOG_ENTRY,
    CHARACTER_INPUT_EXTENSION as CONJ_CHARACTER_INPUT_EXTENSION, DESCRIPTOR as CONJ_DESCRIPTOR,
    ERROR_INTERNAL as CONJ_ERROR_INTERNAL, ERROR_INVALID_INPUT as CONJ_ERROR_INVALID_INPUT,
    EXTENSIONS as CONJ_EXTENSIONS, INTEGER_CAPABILITIES as CONJ_INTEGER_CAPABILITIES,
};
pub use imag::{
    CATALOG_ENTRY as IMAG_CATALOG_ENTRY, DESCRIPTOR as IMAG_DESCRIPTOR,
    ERROR_INTERNAL as IMAG_ERROR_INTERNAL, ERROR_INVALID_INPUT as IMAG_ERROR_INVALID_INPUT,
    INTEGER_CAPABILITIES as IMAG_INTEGER_CAPABILITIES,
};
pub use real::{
    CATALOG_ENTRY as REAL_CATALOG_ENTRY, DESCRIPTOR as REAL_DESCRIPTOR,
    ERROR_INTERNAL as REAL_ERROR_INTERNAL, ERROR_INVALID_INPUT as REAL_ERROR_INVALID_INPUT,
    INTEGER_CAPABILITIES as REAL_INTEGER_CAPABILITIES,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CONJ_CATALOG_ENTRY,
    &REAL_CATALOG_ENTRY,
    &IMAG_CATALOG_ENTRY,
];
