use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/comms/pskmod.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pskmod"),
    slug: Some("pskmod"),
    summary: "Map integer symbols or grouped bits to complex M-PSK samples.",
    description: "`pskmod(X,M)` maps integer symbols to unit-magnitude M-PSK constellation points. Phase offset, Gray or binary ordering, custom symbol mapping, bit input, and output precision are optional.",
    keywords: &["pskmod", "psk", "modulation", "communications", "gray", "binary", "phaseoffset", "constellation", "gpu"],
    related: &["qammod", "scatterplot", "gpuArray", "gather"],
    sections: &[
        BuiltinDocumentationSection { heading: "Symbols and mapping", paragraphs: &[
            "Integer-symbol input must contain finite whole values from 0 through M-1. M must be an integer greater than one. The default phase offset is zero and default symbol order is Gray; 'bin' maps symbol k to phaseoffset + 2*pi*k/M.",
            "A custom symbol-order vector has M distinct whole values from 0 through M-1 and labels constellation points counter-clockwise from the phase offset. InputType='bit' groups rows by log2(M) with the first bit as most significant; it requires a power-of-two M and preserves the remaining channel dimensions. Logical inputs, including scalars, require bit input.",
            "Typed-integer M, phase offset, and custom symbol-order controls are independently gated RunMat extensions. In strict MATLAB mode those extension forms fail before modulation. PlotConstellation=true is recognized but reports an explicit unsupported-operation error.",
        ] },
        BuiltinDocumentationSection { heading: "Precision and execution", paragraphs: &[
            "OutputDataType='single' rounds each real and imaginary coordinate to single; the default is complex double. The output shape follows X, except bit input divides the first dimension by log2(M).",
            "GPU-resident integer-symbol and bit inputs use the owning provider's constellation hooks when available and return complex-interleaved resident storage. Unsupported hooks use the existing host validation and protected re-upload path; malformed provider outputs are rejected. Modulation is a fusion boundary. The operation has no callback, workspace, filesystem, network, randomness, or time effects.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "gray-eight", title: "Default 8-PSK Gray mapping", program: "y = pskmod(0:7,8);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y),[1 8])); assert(all(abs(abs(y)-1)<1e-12));" } },
        BuiltinExample { id: "phase-offset", title: "QPSK with phase offset", program: "y = pskmod(0:3,4,pi/4);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y),[1 4])); assert(abs(y(1)-exp(1i*pi/4))<1e-12);" } },
        BuiltinExample { id: "binary-order", title: "Use binary symbol order", program: "y = pskmod(0:3,4,0,'bin');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(y(1)-1)<1e-12); assert(abs(y(2)-1i)<1e-12); assert(abs(y(3)+1)<1e-12);" } },
        BuiltinExample { id: "bit-input", title: "Group bit rows into QPSK symbols", program: "bits = [0 0; 0 1; 1 0; 1 1]; y = pskmod(bits,4,'InputType','bit');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y),[2 2])); assert(all(abs(abs(y(:))-1)<1e-12));" } },
        BuiltinExample { id: "single-output", title: "Choose complex single output", program: "y = pskmod(0:3,4,'OutputDataType','single');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(y),'single')); assert(isequal(size(y),[1 4]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does pskmod execute on the GPU?", answer: "When the owning provider supports modulation hooks, resident symbol and bit arrays produce resident complex output. An unsupported hook can use a validated host fallback and re-upload." },
        BuiltinDocumentationFaq { question: "Is custom symbol ordering supported?", answer: "Yes. Supply a permutation whose first element labels the point at phaseoffset and whose subsequent elements label points counter-clockwise." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "qammod", target: BuiltinDocumentationLinkTarget::Builtin("qammod") },
        BuiltinDocumentationLink { label: "scatterplot", target: BuiltinDocumentationLinkTarget::Builtin("scatterplot") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "PSK mapping, typed controls, bit grouping, host/provider precision, and invalid input", location: "crates/runmat-runtime/src/builtins/comms/pskmod.rs" }],
        notes: &["Provider execution, fallback, and browser graphics are separate integrated-product verification gates; this source declaration does not claim those gates passed."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
