use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/acceleration/gpu/gather.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Host, provider, container, and output-count tests",
            location: "builtins::acceleration::gpu::gather::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "WGPU native download round trip",
            location: "builtins::acceleration::gpu::gather::tests::gather_wgpu_provider_roundtrip",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`X = gather(A)` copies a provider-resident value to host memory. The result retains the source class, shape, and supported real or complex layout. Host-resident input passes through unchanged.",
            "`[X1, X2, ...] = gather(A1, A2, ...)` gathers each input in call order. The number of requested outputs must match the number of inputs. RunMat mode also descends through cells, structs, and supported object values and replaces nested gpuArray handles with host values.",
            "Gathering does not consume the source handle. It remains valid and resident until the program releases it. Provider download errors remain visible to the caller rather than changing the value's class or precision.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "`gather` is an explicit device-to-host boundary and a fusion sink. Each gpuArray is resolved through its owning provider, so a later provider registration cannot redirect an existing handle to a different backend.",
            "Transfers carry the physical element type and layout. Single and fixed-width integer values are reconstructed from native storage instead of passing through double precision.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "matrix-download",
        title: "Copy a gpuArray matrix to host memory",
        program: "G = gpuArray([1 2 3; 4 5 6]);\nH = gather(G)",
        display_output: Some("H = [1 2 3; 4 5 6]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(H, \"double\"));\nassert(isequal(H, [1 2 3; 4 5 6]));",
        },
    },
    BuiltinExample {
        id: "host-pass-through",
        title: "Pass through data already in host memory",
        program: "x = uint16([10 20 30]);\ny = gather(x)",
        display_output: Some("y = uint16([10 20 30])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, \"uint16\"));\nassert(isequal(y, x));",
        },
    },
    BuiltinExample {
        id: "logical-download",
        title: "Preserve logical class and shape",
        program: "mask = gpuArray(logical([1 0 1 0]));\nhostMask = gather(mask)",
        display_output: Some("hostMask = logical([1 0 1 0])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(hostMask, \"logical\"));\nassert(isequal(hostMask, logical([1 0 1 0])));",
        },
    },
    BuiltinExample {
        id: "nested-cell",
        title: "Gather gpuArray values inside a cell array",
        program: "C = {gpuArray([1 2]), uint8(42)};\nhostC = gather(C)",
        display_output: Some("hostC contains host values [1 2] and uint8(42)"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(hostC{1}, [1 2]));\nassert(isa(hostC{2}, \"uint8\"));\nassert(hostC{2} == uint8(42));",
        },
    },
    BuiltinExample {
        id: "nested-struct",
        title: "Gather gpuArray values in struct fields",
        program: "S = struct(\"data\", gpuArray([8 1 6; 3 5 7; 4 9 2]), \"label\", \"gpu result\");\nS_host = gather(S)",
        display_output: Some("S_host.data is a host matrix and S_host.label is unchanged"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(S_host.data, [8 1 6; 3 5 7; 4 9 2]));\nassert(S_host.label == \"gpu result\");",
        },
    },
    BuiltinExample {
        id: "multiple-outputs",
        title: "Gather multiple inputs into matching outputs",
        program: "A = gpuArray(eye(3));\nB = gpuArray(ones(2, 2));\n[hostA, hostB] = gather(A, B)",
        display_output: Some("hostA is eye(3); hostB is ones(2, 2)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(hostA, eye(3)));\nassert(isequal(hostB, ones(2, 2)));",
        },
    },
    BuiltinExample {
        id: "pipeline-boundary",
        title: "Gather at the end of a GPU pipeline",
        program: "A = gpuArray([0 pi/6 pi/2]);\nB = sin(A) .* 5;\nresult = gather(B)",
        display_output: Some("result = [0 2.5 5]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(result - [0 2.5 5])) < 1e-5);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `gather` modify the original gpuArray?", answer: "No. It returns a host value and leaves the source handle valid and resident." },
    BuiltinDocumentationFaq { question: "What happens to input that is already on the host?", answer: "It passes through unchanged, including its class and shape." },
    BuiltinDocumentationFaq { question: "Are logical and integer classes preserved?", answer: "Yes. Native download metadata reconstructs logical, single, double, and all eight fixed-width integer classes without widening through double." },
    BuiltinDocumentationFaq { question: "Does `gather` recurse into containers?", answer: "In RunMat compatibility mode, it gathers nested gpuArray values in cells, structs, and supported objects. This recursive container behavior is a RunMat extension." },
    BuiltinDocumentationFaq { question: "How do multiple inputs work?", answer: "Request one output for each input: `[X1, X2] = gather(A1, A2)`. The input and output counts must match." },
    BuiltinDocumentationFaq { question: "What happens if no provider owns a gpuArray handle?", answer: "The call returns a provider error. Host-only inputs do not require an acceleration provider." },
    BuiltinDocumentationFaq { question: "Does `gather` free device memory?", answer: "No. Clear or release the source gpuArray when it is no longer needed." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
    BuiltinDocumentationLink {
        label: "gpuDevice",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuDevice"),
    },
    BuiltinDocumentationLink {
        label: "sum",
        target: BuiltinDocumentationLinkTarget::Builtin("sum"),
    },
    BuiltinDocumentationLink {
        label: "mean",
        target: BuiltinDocumentationLinkTarget::Builtin("mean"),
    },
    BuiltinDocumentationLink {
        label: "arrayfun",
        target: BuiltinDocumentationLinkTarget::Builtin("arrayfun"),
    },
    BuiltinDocumentationLink {
        label: "gpuInfo",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuInfo"),
    },
    BuiltinDocumentationLink {
        label: "pagefun",
        target: BuiltinDocumentationLinkTarget::Builtin("pagefun"),
    },
];

pub(in super::super) const GATHER_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gather"),
    slug: Some("gather"),
    summary: "Copy provider-resident data back to host memory.",
    description: "`gather` reconstructs host values from gpuArray storage while preserving class and shape. Host values pass through unchanged, and RunMat mode can gather gpuArray values nested in containers.",
    keywords: &["gather", "gpuArray", "download", "host copy", "accelerate", "residency"],
    related: &["arrayfun", "gpuArray", "gpuDevice", "gpuInfo", "mean", "pagefun", "sum"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
