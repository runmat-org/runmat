use super::text_inference::{self, TextConversion};
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

const D: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Real numeric, logical, or character values; resident numeric input may be gathered.",
};
const WIDTH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "minDigits",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("0"),
    description: "Minimum hexadecimal row width as a nonnegative numeric or logical scalar.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Uppercase hexadecimal character rows.",
}];
const ONE: [BuiltinParamDescriptor; 1] = [D];
const TWO: [BuiltinParamDescriptor; 2] = [D, WIDTH];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "s = dec2hex(D)",
        inputs: &ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = dec2hex(D, minDigits)",
        inputs: &TWO,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.DEC2HEX.INPUT",
        identifier: None,
        when: "D is unsupported, fractional, nonfinite, or outside the supported integer range.",
        message: "dec2hex: D must contain finite real integer values in range",
    },
    BuiltinErrorDescriptor {
        code: "RM.DEC2HEX.WIDTH",
        identifier: None,
        when: "minDigits is not a nonnegative integer-valued numeric or logical scalar.",
        message: "dec2hex: minDigits must be a nonnegative integer-valued scalar",
    },
    BuiltinErrorDescriptor {
        code: "RM.DEC2HEX.OUTPUT",
        identifier: None,
        when: "The output exceeds the character allocation safety limit.",
        message: "dec2hex: requested output is too large",
    },
];
pub const DEC2HEX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "D", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Read exact integer storage, preserving signed source width for negative two's-complement formatting." }];
const INTEGER_WIDTH: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "minDigits",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "An exact fixed-width nonnegative integer scalar controls minimum row width.",
}];
pub const DEC2HEX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "s = dec2hex(integer_D, minDigits)",
        inputs: &INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "The result is host character data; exact integer inputs never round through double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "s = dec2hex(D, integer_minDigits)",
        inputs: &INTEGER_WIDTH,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Width is decoded exactly before any resident input download.",
    },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/comms/binary_conversion.rs") };
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("dec2hex"), slug: Some("dec2hex"), summary: "Convert decimal integers to uppercase hexadecimal character rows.",
    description: "`dec2hex(D)` formats every element of `D` in column-major order as a hexadecimal character row. `dec2hex(D,minDigits)` pads rows to a minimum width.", keywords: &["dec2hex", "decimal", "hexadecimal", "two's complement"], related: &["hex2dec", "dec2bin"],
    sections: &[
        BuiltinDocumentationSection { heading: "Values and width", paragraphs: &["Real numeric input must contain finite integer values; logical and character input are also accepted. Negative values use two's complement in the smallest fitting 8-, 16-, 32-, or 64-bit width. Typed signed integers preserve their source width. Extra padding sign-extends with `F`.", "The output has one character row per input element, including N-D numeric input in column-major order. `minDigits` accepts a finite nonnegative integer-valued numeric or logical scalar; false means zero minimum digits and true means one. Output size is bounded before formatting."] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &["Resident numeric input gathers to the host after argument admission. Formatting is a fusion boundary without a device kernel; it has no callbacks or ambient service dependencies."] },
    ],
    examples: &[
        BuiltinExample { id: "scalar", title: "Hexadecimal digits", program: "s = dec2hex(1023)", display_output: Some("s = '3FF'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(s,'3FF'));" } },
        BuiltinExample { id: "padded", title: "Pad to six digits", program: "s = dec2hex(1023,6)", display_output: Some("s = '0003FF'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(s,'0003FF'));" } },
        BuiltinExample { id: "rows", title: "Convert a vector", program: "s = dec2hex([1023 122 14])", display_output: Some("s = ['3FF'; '07A'; '00E']"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(s),[3 3])); assert(strcmp(s(2,:),'07A'));" } },
        BuiltinExample { id: "negative", title: "Two's complement", program: "s = dec2hex(-16)", display_output: Some("s = 'F0'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(s,'F0'));" } },
    ], example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Are fractions accepted?", answer: "No. Floating-point input must be an exact integer value." }, BuiltinDocumentationFaq { question: "Are letters uppercase?", answer: "Yes, hexadecimal A through F are uppercase." }],
    links: &[BuiltinDocumentationLink { label: "hex2dec", target: BuiltinDocumentationLinkTarget::Builtin("hex2dec") }, BuiltinDocumentationLink { label: "dec2bin", target: BuiltinDocumentationLinkTarget::Builtin("dec2bin") }, SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, vector, width, negative, exact integer, and error tests", location: "crates/runmat-runtime/src/builtins/comms/binary_conversion.rs" }], notes: &["Resident numeric inputs materialize to a host character result."] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    text_inference::infer(request, entry, TextConversion::Dec2Hex)
}
pub const DEC2HEX_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "dec2hex" },
    category: "comms/conversion",
    documentation: DOCUMENTATION,
    descriptor: &DEC2HEX_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            infer as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &DEC2HEX_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
