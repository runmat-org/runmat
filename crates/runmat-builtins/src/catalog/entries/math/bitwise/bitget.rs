use super::{BITWISE_ERROR_INVALID_INPUT, BITWISE_ERROR_SIZE_MISMATCH, DIRECT_BIT_EXTENSIONS};
use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected zero-or-one bit values in the class of A.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued data input.",
    },
    BuiltinParamDescriptor {
        name: "bit",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based bit position.",
    },
];
const ASSUMED_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued data input.",
    },
    BuiltinParamDescriptor {
        name: "bit",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based bit position.",
    },
    BuiltinParamDescriptor {
        name: "assumedtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Integer class used to interpret double input A.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = bitget(A, bit)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "B = bitget(A, bit, assumedtype)",
        inputs: &ASSUMED_INPUTS,
        outputs: &OUTPUTS,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [BITWISE_ERROR_INVALID_INPUT, BITWISE_ERROR_SIZE_MISMATCH];
pub const BITGET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every fixed-width integer data class supplies exact signed or unsigned storage and determines output class and residency.",
    },
    BuiltinIntegerInputCapability {
        name: "bit",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is accepted for finite positive one-based bit positions; this control does not select output class.",
    },
];
pub const BITGET_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = bitget(integer_A, integer_bit)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "A and bit use scalar expansion or exactly matching nonscalar sizes. Resident data uses exact fallback and returns to A's owner.",
    }];
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`B = bitget(A, bit)` returns the selected one-based bit from each element of `A`. The result contains zero or one, retains the class of `A`, and uses scalar expansion or exactly matching nonscalar sizes.",
            "`B = bitget(A, bit, assumedtype)` selects the signed or unsigned width used for integer-valued `double` input. Each bit position must be a finite positive integer within that width.",
            "All eight fixed-width integer classes are accepted for both data and bit-position inputs. The bit-position class does not affect output class.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resident values",
        paragraphs: &[
            "Supported resident data gathers exactly and the result returns to the owner. The bit position is a structural control. Resident forms outside the documented domain and resident `assumedtype` calls are gated RunMat extensions.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "positions",
    title: "Read several bit positions",
    program: "a = uint8([5 6 7]);\nb = bitget(a, uint8([1 2 3]))",
    display_output: Some("b = uint8([1 1 1])"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(b, 'uint8')); assert(isequal(b, uint8([1 1 1])));",
    },
}];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Bit-position runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/bitwise/bitget.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "All classes, exact position rules, sparse values, and resident fallback",
        location: "crates/runmat-runtime/src/builtins/math/bitwise/engine/tests/position.rs",
    }],
    notes: &[],
};
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bitget"),
    slug: Some("bitget"),
    summary: "Read one-based bit positions from integer-valued values.",
    description: "`bitget` extracts selected bits while retaining the data input's numeric class and compatible shape.",
    keywords: &["bitget", "bitwise", "bit", "position", "integer", "gpuArray"],
    related: &["bitset", "bitcmp", "bitshift"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "bitset",
            target: BuiltinDocumentationLinkTarget::Builtin("bitset"),
        },
        BuiltinDocumentationLink {
            label: "bitcmp",
            target: BuiltinDocumentationLinkTarget::Builtin("bitcmp"),
        },
    ],
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const BITGET_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "bitget" },
    category: "math/bitwise",
    documentation: DOCUMENTATION,
    descriptor: &BITGET_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Bitwise(
            BitwiseInferenceRule::Get,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::Elementwise,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &DIRECT_BIT_EXTENSIONS,
    integer_capabilities: &BITGET_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) fn extend_entries(values: &mut Vec<&'static BuiltinCatalogEntry>) {
    values.push(&BITGET_CATALOG_ENTRY);
}
