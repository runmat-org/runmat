use super::text_inference::{self, TextConversion};
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "binStr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Binary character rows, string array, or cell array of text.",
}];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Decimal double value for each input text item.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "D = bin2dec(binStr)",
    inputs: &INPUT,
    outputs: &OUTPUT,
}];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.BIN2DEC.INPUT",
        identifier: None,
        when: "Input is not text or a cell contains a nontext value.",
        message: "bin2dec: input must be text",
    },
    BuiltinErrorDescriptor {
        code: "RM.BIN2DEC.DIGITS",
        identifier: None,
        when: "Text is empty, malformed, too long, or contains a digit outside base two.",
        message: "bin2dec: input text contains invalid digits",
    },
    BuiltinErrorDescriptor {
        code: "RM.BIN2DEC.RANGE",
        identifier: None,
        when: "A suffix width overflows or the result exceeds the exact supported decimal range.",
        message: "bin2dec: result exceeds the maximum exact integer supported by RunMat",
    },
];
pub const BIN2DEC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/comms/binary_conversion.rs") };
const MATLAB_REFERENCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "MATLAB bin2dec reference",
    target: BuiltinDocumentationLinkTarget::External(
        "https://www.mathworks.com/help/matlab/ref/bin2dec.html",
    ),
};
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("bin2dec"), slug: Some("bin2dec"), summary: "Convert binary text to decimal double values.",
    description: "`bin2dec(binStr)` reads each binary character row, string element, or text cell as a decimal value. A character matrix and cell array produce a column vector; string arrays preserve their shape.",
    keywords: &["bin2dec", "binary", "decimal", "two's complement"], related: &["dec2bin", "hex2dec"],
    sections: &[
        BuiltinDocumentationSection { heading: "Text forms", paragraphs: &["Character vectors and string scalars produce scalar double output. Character matrices parse one row at a time. Each cell element must be one character row or a singleton string. A string array with exactly one item returns a scalar; an empty string array retains its shape. Whitespace around each item is trimmed.", "Prefixes `0b` and `0B` and signed/unsigned literal suffixes (`s8` through `s64`, `u8` through `u64`) are also MATLAB-supported forms since R2020a. Signed suffixes interpret the bits in two's complement; values must fit the suffix width. RunMat's current policy rejects results above 2^52 in magnitude, whereas MATLAB warns at or above `flintmax` and may return an inexact wide double."] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &["Text parsing runs on the host and is a fusion boundary. No resident text kernel, callbacks, workspace, I/O, randomness, or time service is involved."] },
    ],
    examples: &[
        BuiltinExample { id: "scalar", title: "Parse a binary character vector", program: "d = bin2dec('10111')", display_output: Some("d = 23"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d,23));" } },
        BuiltinExample { id: "rows", title: "Parse character rows", program: "d = bin2dec(['1111111111'; '0001111010'; '0000001110'])", display_output: Some("d = [1023; 122; 14]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d,[1023;122;14])); assert(isequal(size(d),[3 1]));" } },
        BuiltinExample { id: "strings", title: "Preserve a string array's shape", program: "d = bin2dec([\"1111111111\" \"1111010\" \"1110\"])", display_output: Some("d = [1023 122 14]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d,[1023 122 14])); assert(isequal(size(d),[1 3]));" } },
        BuiltinExample { id: "signed-suffix", title: "Interpret a signed suffix", program: "d = bin2dec('0b11111111s8')", display_output: Some("d = -1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d,-1));" } },
    ], example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Are binary prefixes accepted?", answer: "Yes, `0b` and `0B` are accepted." }, BuiltinDocumentationFaq { question: "What does `s8` mean?", answer: "It interprets eight bits as a signed two's-complement value." }, BuiltinDocumentationFaq { question: "What happens above 2^52?", answer: "The conversion rejects values that exceed the supported exact decimal range." }],
    links: &[BuiltinDocumentationLink { label: "dec2bin", target: BuiltinDocumentationLinkTarget::Builtin("dec2bin") }, BuiltinDocumentationLink { label: "hex2dec", target: BuiltinDocumentationLinkTarget::Builtin("hex2dec") }, MATLAB_REFERENCE, SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Text shapes, prefixes, suffixes, exact range, and invalid digits", location: "crates/runmat-runtime/src/builtins/comms/binary_conversion.rs" }], notes: &["The MATLAB reference documents prefix and signed-suffix support since R2020a; RunMat's >2^52 rejection is a stricter existing numerical policy."] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    text_inference::infer(request, entry, TextConversion::Bin2Dec)
}
pub const BIN2DEC_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "bin2dec" },
    category: "comms/conversion",
    documentation: DOCUMENTATION,
    descriptor: &BIN2DEC_DESCRIPTOR,
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
