use super::text_inference::{self, TextConversion};
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "hexStr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hexadecimal character rows, string array, or cell array of text.",
}];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Decimal double value for each input text item.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "D = hex2dec(hexStr)",
    inputs: &INPUT,
    outputs: &OUTPUT,
}];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.HEX2DEC.INPUT",
        identifier: None,
        when: "Input is not text or a cell contains a nontext value.",
        message: "hex2dec: input must be text",
    },
    BuiltinErrorDescriptor {
        code: "RM.HEX2DEC.DIGITS",
        identifier: None,
        when: "Text is empty, malformed, too long, or contains a digit outside base sixteen.",
        message: "hex2dec: input text contains invalid digits",
    },
    BuiltinErrorDescriptor {
        code: "RM.HEX2DEC.RANGE",
        identifier: None,
        when: "A suffix width overflows or the result exceeds the exact supported decimal range.",
        message: "hex2dec: result exceeds the maximum exact integer supported by RunMat",
    },
];
pub const HEX2DEC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/comms/binary_conversion.rs") };
const MATLAB_REFERENCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "MATLAB hex2dec reference",
    target: BuiltinDocumentationLinkTarget::External(
        "https://www.mathworks.com/help/matlab/ref/hex2dec.html",
    ),
};
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("hex2dec"), slug: Some("hex2dec"), summary: "Convert hexadecimal text to decimal double values.",
    description: "`hex2dec(hexStr)` reads each hexadecimal character row, string element, or text cell as a decimal value. A character matrix and cell array produce a column vector; string arrays preserve their shape.",
    keywords: &["hex2dec", "hexadecimal", "decimal", "two's complement"], related: &["dec2hex", "bin2dec"],
    sections: &[
        BuiltinDocumentationSection { heading: "Text forms", paragraphs: &["Character vectors and string scalars produce scalar double output. Character matrices parse one row at a time. Each cell element must be one character row or a singleton string. A string array with exactly one item returns a scalar; an empty string array retains its shape. Whitespace around each item is trimmed; mixed-case hexadecimal digits are accepted.", "Prefixes `0x` and `0X` and signed/unsigned literal suffixes (`s8` through `s64`, `u8` through `u64`) are also MATLAB-supported forms since R2020a. Signed suffixes interpret the digits in two's complement; values must fit the suffix width. RunMat's current policy rejects results above 2^52 in magnitude, whereas MATLAB warns at or above `flintmax` and may return an inexact wide double."] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &["Text parsing runs on the host and is a fusion boundary. No resident text kernel, callbacks, workspace, I/O, randomness, or time service is involved."] },
    ],
    examples: &[
        BuiltinExample { id: "scalar", title: "Parse hexadecimal text", program: "d = hex2dec('3FF')", display_output: Some("d = 1023"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d,1023));" } },
        BuiltinExample { id: "strings", title: "Preserve a string array's shape", program: "d = hex2dec([\"3FF\" \"7A\" \"E\"])", display_output: Some("d = [1023 122 14]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d,[1023 122 14])); assert(isequal(size(d),[1 3]));" } },
        BuiltinExample { id: "prefix", title: "Accept hexadecimal prefix", program: "d = hex2dec('0xFF')", display_output: Some("d = 255"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d,255));" } },
        BuiltinExample { id: "signed-suffix", title: "Interpret a signed suffix", program: "d = hex2dec('0xFFs8')", display_output: Some("d = -1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d,-1));" } },
    ], example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Are hexadecimal prefixes accepted?", answer: "Yes, `0x` and `0X` are accepted." }, BuiltinDocumentationFaq { question: "What does `s8` mean?", answer: "It interprets eight bits as a signed two's-complement value." }, BuiltinDocumentationFaq { question: "What happens above 2^52?", answer: "The conversion rejects values that exceed the supported exact decimal range." }],
    links: &[BuiltinDocumentationLink { label: "dec2hex", target: BuiltinDocumentationLinkTarget::Builtin("dec2hex") }, BuiltinDocumentationLink { label: "bin2dec", target: BuiltinDocumentationLinkTarget::Builtin("bin2dec") }, MATLAB_REFERENCE, SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Text shapes, prefixes, suffixes, exact range, and invalid digits", location: "crates/runmat-runtime/src/builtins/comms/binary_conversion.rs" }], notes: &["The MATLAB reference documents prefix and signed-suffix support since R2020a; RunMat's >2^52 rejection is a stricter existing numerical policy."] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    text_inference::infer(request, entry, TextConversion::Hex2Dec)
}
pub const HEX2DEC_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "hex2dec" },
    category: "comms/conversion",
    documentation: DOCUMENTATION,
    descriptor: &HEX2DEC_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            infer as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MayThrow],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::Host,
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
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};
