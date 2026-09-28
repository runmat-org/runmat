use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/comms/qammod.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("qammod"),
    slug: Some("qammod"),
    summary: "Map integer symbols or grouped bits to rectangular QAM points.",
    description: "`qammod(X,M)` maps whole-valued symbols or grouped bits to complex QAM points. Gray and binary orders, a custom symbol permutation, unit-average-power scaling, and output class can be selected.",
    keywords: &["qammod", "qam", "modulation", "communications", "gray", "binary", "constellation", "gpu"],
    related: &["pskmod", "scatterplot", "gpuArray", "gather"],
    sections: &[
        BuiltinDocumentationSection { heading: "Constellation and controls", paragraphs: &[
            "X must contain finite whole symbols from 0 through M-1. M must be a power of two greater than one. Square orders use symmetric odd in-phase and quadrature amplitudes; non-square powers of two use a rectangular grid with the longer in-phase axis.",
            "Gray order is the default and applies to each grid axis. 'bin' uses binary order; a custom permutation labels constellation points. InputType='bit' groups rows in log2(M) chunks per channel, first bit most significant. Logical inputs, including scalars, require bit input. UnitAveragePower=true scales the constellation to average unit power.",
            "PlotConstellation=true is recognized but explicitly unsupported. The output precision defaults to single only when X has single storage, otherwise double; OutputDataType selects 'single' or 'double'.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &[
            "Supported resident symbol and bit arrays use the owning acceleration provider's modulation hooks and produce complex-interleaved resident output. Unsupported hooks use host validation and protected re-upload; malformed provider output is rejected. Modulation terminates fusion.",
            "This pure operation has no callback, workspace, filesystem, network, randomness, or time effects. Explicit GPU input keeps device intent through the supported hook or the fallback upload.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "gray-sixteen", title: "Default 16-QAM Gray mapping", program: "y = qammod(0:15,16);", display_output: Some("y = [-3+3i -3+1i -3-3i -3-1i -1+3i -1+1i -1-3i -1-1i 3+3i 3+1i 3-3i 3-1i 1+3i 1+1i 1-3i 1-1i]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "expected = [-3+3i -3+1i -3-3i -3-1i -1+3i -1+1i -1-3i -1-1i 3+3i 3+1i 3-3i 3-1i 1+3i 1+1i 1-3i 1-1i]; assert(isequal(size(y),[1 16])); assert(all(abs(y-expected)<1e-12));" } },
        BuiltinExample { id: "binary-order", title: "Use binary order", program: "y = qammod(0:4,16,'bin');", display_output: Some("The first points begin -3+3i, -3+1i, -3-1i, -3-3i."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y),[1 5])); assert(abs(y(1)-(-3+3i))<1e-12); assert(abs(y(2)-(-3+1i))<1e-12);" } },
        BuiltinExample { id: "unit-power", title: "Normalize average constellation power", program: "y = qammod(0:15,16,'UnitAveragePower',true); p = mean(abs(y).^2);", display_output: Some("p = 1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(p-1)<1e-12);" } },
        BuiltinExample { id: "bit-input", title: "Group bit rows into QAM symbols", program: "bits = [0 1;0 1;0 1;0 1]; y = qammod(bits,16,'InputType','bit');", display_output: Some("y = [-3+3i 1-1i]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y),[1 2])); assert(all(abs(y-[-3+3i 1-1i])<1e-12));" } },
        BuiltinExample { id: "single-default", title: "Preserve single input precision", program: "y = qammod(single(0:3),4);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(y),'single')); assert(isequal(size(y),[1 4]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does qammod execute on the GPU?", answer: "When the owning provider has modulation hooks, resident symbol and bit inputs produce resident complex output. Unsupported hooks can use host validation and re-upload." },
        BuiltinDocumentationFaq { question: "Is bit input supported?", answer: "Yes. InputType='bit' accepts numeric zero/one or logical arrays and groups log2(M) rows per symbol for each channel." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "pskmod", target: BuiltinDocumentationLinkTarget::Builtin("pskmod") },
        BuiltinDocumentationLink { label: "scatterplot", target: BuiltinDocumentationLinkTarget::Builtin("scatterplot") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "QAM mapping, bit grouping, typed symbols, host/provider precision, and invalid input", location: "crates/runmat-runtime/src/builtins/comms/qammod.rs" }],
        notes: &["Provider execution and browser runtime remain separate integrated-product verification gates."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
