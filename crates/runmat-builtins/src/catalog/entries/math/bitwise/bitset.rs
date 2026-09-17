use super::{BITWISE_ERROR_INVALID_INPUT, BITWISE_ERROR_SIZE_MISMATCH, DIRECT_BIT_EXTENSIONS};
use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated values in the class of A.",
}];
const A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Integer-valued data input.",
};
const BIT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "bit",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based bit position.",
};
const V: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "V",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Zero clears the bit; a finite nonzero value sets it.",
};
const ASSUMED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "assumedtype",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Integer class used to interpret double input A.",
};
const INPUTS: [BuiltinParamDescriptor; 2] = [A, BIT];
const VALUE_INPUTS: [BuiltinParamDescriptor; 3] = [A, BIT, V];
const ASSUMED_INPUTS: [BuiltinParamDescriptor; 3] = [A, BIT, ASSUMED];
const VALUE_ASSUMED_INPUTS: [BuiltinParamDescriptor; 4] = [A, BIT, V, ASSUMED];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "C = bitset(A, bit)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitset(A, bit, V)",
        inputs: &VALUE_INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitset(A, bit, assumedtype)",
        inputs: &ASSUMED_INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitset(A, bit, V, assumedtype)",
        inputs: &VALUE_ASSUMED_INPUTS,
        outputs: &OUTPUTS,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [BITWISE_ERROR_INVALID_INPUT, BITWISE_ERROR_SIZE_MISMATCH];
pub const BITSET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The data class supplies exact storage and determines output class and residency.",
    },
    BuiltinIntegerInputCapability {
        name: "bit",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is accepted for finite positive one-based positions.",
    },
    BuiltinIntegerInputCapability {
        name: "V",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is accepted for zero-or-one replacement controls.",
    },
];
pub const BITSET_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = bitset(integer_A, integer_bit, integer_V)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "A, bit, and V use scalar expansion or exactly matching nonscalar sizes. Resident data uses exact fallback and returns to A's owner.",
    }];
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`C = bitset(A, bit)` sets the selected one-based bit in every element of `A`. `C = bitset(A, bit, V)` sets it when `V` is nonzero and clears it when `V` is zero.",
            "A, bit, and V use scalar expansion or exactly matching nonscalar sizes. The result retains A's class. All eight fixed-width integer classes are accepted for data and controls.",
            "An optional `assumedtype` selects the signed or unsigned width for integer-valued `double` input. Positions must be finite positive integers within that width; V must be finite.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resident values",
        paragraphs: &[
            "Supported resident data gathers exactly and the updated result returns to the owner. Position and replacement values are controls. Broader resident forms and resident `assumedtype` calls are gated RunMat extensions.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "set-and-clear",
    title: "Set and clear selected bits",
    program: "a = uint8([0 7]);\nc = bitset(a, uint8([1 2]), uint8([1 0]))",
    display_output: Some("c = uint8([1 5])"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(c, 'uint8')); assert(isequal(c, uint8([1 5])));",
    },
}];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Bit-position runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/bitwise/bitset.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "All classes, set/clear controls, sparse behavior, exact shapes, and resident fallback",
        location: "crates/runmat-runtime/src/builtins/math/bitwise/engine/tests/position.rs",
    }],
    notes: &[],
};
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bitset"),
    slug: Some("bitset"),
    summary: "Set or clear one-based bit positions in integer-valued values.",
    description: "`bitset` updates selected bits while retaining the data input's numeric class and compatible shape.",
    keywords: &["bitset", "bitwise", "bit", "position", "integer", "gpuArray"],
    related: &["bitget", "bitcmp", "bitshift"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "bitget",
            target: BuiltinDocumentationLinkTarget::Builtin("bitget"),
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
pub const BITSET_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "bitset" },
    category: "math/bitwise",
    documentation: DOCUMENTATION,
    descriptor: &BITSET_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Bitwise(
            BitwiseInferenceRule::Set,
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
    integer_capabilities: &BITSET_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) fn extend_entries(values: &mut Vec<&'static BuiltinCatalogEntry>) {
    values.push(&BITSET_CATALOG_ENTRY);
}
