use super::{BITWISE_ERROR_INVALID_INPUT, BITWISE_ERROR_SIZE_MISMATCH, DIRECT_BIT_EXTENSIONS};
use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bitwise complement with the input class and shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Integer-valued input whose bits are complemented.",
}];
const ASSUMED_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input whose bits are complemented.",
    },
    BuiltinParamDescriptor {
        name: "assumedtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Integer class used to interpret double input.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = bitcmp(A)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitcmp(A, assumedtype)",
        inputs: &ASSUMED_INPUTS,
        outputs: &OUTPUTS,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [BITWISE_ERROR_INVALID_INPUT, BITWISE_ERROR_SIZE_MISMATCH];
pub const BITCMP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every fixed-width integer class is complemented within its own width.",
}];
pub const BITCMP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = bitcmp(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host execution covers all eight classes. Documented resident input is uint8, uint16, or uint32 and returns to the owning provider.",
    }];
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`bitcmp(A)` complements every bit of each integer element and preserves class and shape. For `double` input, the default interpretation uses 64 unsigned bits.",
            "`bitcmp(A, assumedtype)` uses the named signed or unsigned integer width for integer-valued `double` input. Native integer input already supplies its class and must agree with an explicit assumed type.",
            "Bit positions beyond the selected width do not participate. Unsupported, fractional, out-of-range, complex, or incompatible inputs produce structured errors.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resident values",
        paragraphs: &[
            "Documented resident input is `uint8`, `uint16`, or `uint32`. Exact host fallback restores the result to the owning provider. Broader resident classes and resident `assumedtype` calls are RunMat extensions.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "uint8",
        title: "Complement an eight-bit value",
        program: "c = bitcmp(uint8(15))",
        display_output: Some("c = uint8(240)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(c, 'uint8')); assert(c == uint8(240));",
        },
    },
    BuiltinExample {
        id: "assumed-width",
        title: "Choose a width for double input",
        program: "c = bitcmp(15, 'uint8')",
        display_output: Some("c = 240"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(c, 'double')); assert(c == 240);",
        },
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "bitget",
        target: BuiltinDocumentationLinkTarget::Builtin("bitget"),
    },
    BuiltinDocumentationLink {
        label: "bitset",
        target: BuiltinDocumentationLinkTarget::Builtin("bitset"),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Complement runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/bitwise/bitcmp.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "All fixed-width classes, assumed widths, sparse values, and resident fallback",
        location: "crates/runmat-runtime/src/builtins/math/bitwise/engine/tests/complement.rs",
    }],
    notes: &[],
};
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bitcmp"),
    slug: Some("bitcmp"),
    summary: "Compute the bitwise complement of integer-valued values.",
    description: "`bitcmp` complements each element within its integer storage width.",
    keywords: &["bitcmp", "bitwise", "complement", "integer", "gpuArray"],
    related: &["bitget", "bitset", "bitshift"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const BITCMP_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "bitcmp" },
    category: "math/bitwise",
    documentation: DOCUMENTATION,
    descriptor: &BITCMP_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Bitwise(
            BitwiseInferenceRule::Complement,
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
        distributed: BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &DIRECT_BIT_EXTENSIONS,
    integer_capabilities: &BITCMP_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
