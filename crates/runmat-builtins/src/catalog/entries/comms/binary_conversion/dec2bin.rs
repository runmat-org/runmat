use super::text_inference::{self, TextConversion};
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

const INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
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
    description: "Minimum binary row width as a nonnegative numeric or logical scalar.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One binary character row per input element.",
}];
const ONE: [BuiltinParamDescriptor; 1] = [INPUT];
const TWO: [BuiltinParamDescriptor; 2] = [INPUT, WIDTH];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "s = dec2bin(D)",
        inputs: &ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = dec2bin(D, minDigits)",
        inputs: &TWO,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.DEC2BIN.INPUT",
        identifier: None,
        when: "D is unsupported, nonfinite, or outside the supported integer range.",
        message: "dec2bin: D must contain finite real values in range",
    },
    BuiltinErrorDescriptor {
        code: "RM.DEC2BIN.WIDTH",
        identifier: None,
        when: "minDigits is not a nonnegative integer-valued numeric or logical scalar.",
        message: "dec2bin: minDigits must be a nonnegative integer-valued scalar",
    },
    BuiltinErrorDescriptor {
        code: "RM.DEC2BIN.OUTPUT",
        identifier: None,
        when: "The output exceeds the character allocation safety limit.",
        message: "dec2bin: requested output is too large",
    },
];
pub const DEC2BIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
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
pub const DEC2BIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "s = dec2bin(integer_D, minDigits)", inputs: &INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The result is a host character array; numeric inputs gather after admission and exact integers never round through double." },
    BuiltinIntegerCapabilityDescriptor { form: "s = dec2bin(D, integer_minDigits)", inputs: &INTEGER_WIDTH, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Width is decoded exactly before any resident input download." },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/comms/binary_conversion.rs") };
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("dec2bin"), slug: Some("dec2bin"),
    summary: "Convert decimal values to binary character rows.",
    description: "`dec2bin(D)` formats every element of `D` in column-major order as a binary character row. `dec2bin(D,minDigits)` zero-pads positive rows to the requested minimum width.",
    keywords: &["dec2bin", "decimal", "binary", "two's complement"], related: &["bin2dec", "dec2hex"],
    sections: &[
        BuiltinDocumentationSection { heading: "Values and width", paragraphs: &["Real double and single values are floored before conversion. Logical and character input are accepted. Signed negative values use two's complement in the smallest fitting 8-, 16-, 32-, or 64-bit width; typed signed integers retain their storage width. Extra width sign-extends negative rows with ones.", "`minDigits` is a finite nonnegative integer-valued numeric or logical scalar; false requests zero minimum digits and true requests one. The host character output has one row per input element, including N-D numeric input in column-major order. Oversized outputs reject before formatting."] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &["Resident numeric data are gathered to the host after argument admission; text formatting is a fusion boundary without a device kernel. No callback, workspace, filesystem, network, UI, random, or clock service is used."] },
    ],
    examples: &[
        BuiltinExample { id: "scalar", title: "Binary digits", program: "s = dec2bin(23)", display_output: Some("s = '10111'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(s,'10111'));" } },
        BuiltinExample { id: "padded", title: "Pad to eight digits", program: "s = dec2bin(23,8)", display_output: Some("s = '00010111'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(s,'00010111'));" } },
        BuiltinExample { id: "rows", title: "Convert a vector", program: "s = dec2bin([1023 122 14])", display_output: Some("s = ['1111111111'; '0001111010'; '0000001110']"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(s),[3 10])); assert(strcmp(s(2,:),'0001111010'));" } },
        BuiltinExample { id: "negative", title: "Two's complement", program: "s = dec2bin(-16)", display_output: Some("s = '11110000'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(s,'11110000'));" } },
    ], example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does dec2bin round fractions?", answer: "It floors real floating-point inputs before formatting." }, BuiltinDocumentationFaq { question: "How are negative values represented?", answer: "They are sign-extended two's-complement rows." }],
    links: &[BuiltinDocumentationLink { label: "bin2dec", target: BuiltinDocumentationLinkTarget::Builtin("bin2dec") }, BuiltinDocumentationLink { label: "dec2hex", target: BuiltinDocumentationLinkTarget::Builtin("dec2hex") }, SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, vector, width, negative, exact integer, and error tests", location: "crates/runmat-runtime/src/builtins/comms/binary_conversion.rs" }], notes: &["Resident numeric inputs materialize to a host character result."] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    text_inference::infer(request, entry, TextConversion::Dec2Bin)
}
pub const DEC2BIN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "dec2bin" },
    category: "comms/conversion",
    documentation: DOCUMENTATION,
    descriptor: &DEC2BIN_DESCRIPTOR,
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
    integer_capabilities: &DEC2BIN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
