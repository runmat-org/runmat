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
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/acceleration/gpu/gpuarray.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Upload, conversion, shape, compatibility, and provider tests",
            location: "builtins::acceleration::gpu::gpuarray::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Native numeric transfer tests",
            location: "runmat-accelerate provider tests",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`G = gpuArray(X)` uploads a numeric or logical value to the active acceleration provider. The resulting handle records its owner, physical element type, shape, layout, and explicit device-placement origin. Calling `gpuArray` on an existing gpuArray without conversion returns the same resident value.",
            "RunMat compatibility mode also accepts explicit class selectors, reshape dimensions, and `\"like\", prototype`. Class conversion creates a new resident buffer and leaves an existing source handle valid. Reshaping follows column-major order and must preserve the element count.",
            "Real and supported complex values retain their native class during transfer. Character and string uploads are RunMat extensions. A provider must support the requested physical storage; unsupported storage and transfer failures return an error instead of silently widening the value.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "`gpuArray` is an explicit host-to-device boundary. Later provider-capable operations can consume the returned handle without gathering it first. Call `gather` when host data is required.",
            "RunMat can also place eligible values on an accelerator automatically. Explicit `gpuArray` placement is retained as user intent: unsupported operations do not silently discard that placement by moving the value to the host.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "matrix-upload",
        title: "Upload a matrix for elementwise GPU work",
        program: "A = [1 2 3; 4 5 6];\nG = gpuArray(A);\nout = gather(sin(G))",
        display_output: Some("out is sin(A) with the same 2-by-3 shape"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(out), [2 3]));\nassert(max(abs(out(:) - sin(A(:)))) < 1e-6);",
        },
    },
    BuiltinExample {
        id: "single-conversion",
        title: "Convert to single precision while uploading",
        program: "pi_single = gpuArray(pi, \"single\");\nhost = gather(pi_single)",
        display_output: Some("host is the single-precision value nearest pi"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(host, \"single\"));\nassert(abs(double(host) - pi) < 2e-7);",
        },
    },
    BuiltinExample {
        id: "logical-conversion",
        title: "Create logical storage during upload",
        program: "mask = gpuArray([0 2 -5 0], \"logical\");\nhost = gather(mask)",
        display_output: Some("host = logical([0 1 1 0])"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(host, \"logical\"));\nassert(isequal(host, logical([0 1 1 0])));",
        },
    },
    BuiltinExample {
        id: "like-prototype",
        title: "Match a resident prototype's class",
        program: "template = gpuArray(true(2, 2));\nvalues = gpuArray([10 20; 30 40], \"like\", template);\nhost = gather(values)",
        display_output: Some("host = logical([1 1; 1 1])"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(host, \"logical\"));\nassert(isequal(host, logical([10 20; 30 40])));",
        },
    },
    BuiltinExample {
        id: "reshape-upload",
        title: "Reshape in column-major order during upload",
        program: "flat = 1:6;\nG = gpuArray(flat, 2, 3);\nhost = gather(G)",
        display_output: Some("host = [1 3 5; 2 4 6]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(host), [2 3]));\nassert(isequal(host, [1 3 5; 2 4 6]));",
        },
    },
    BuiltinExample {
        id: "resident-identity",
        title: "Keep an existing gpuArray resident",
        program: "G = gpuArray([1 2 3]);\nH = gpuArray(G);\nhost = gather(H)",
        display_output: Some("host = [1 2 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(host, [1 2 3]));\nassert(isequal(gather(G), [1 2 3]));",
        },
    },
    BuiltinExample {
        id: "resident-class-conversion",
        title: "Convert a resident value without consuming its source",
        program: "G = gpuArray(uint16([1 2; 3 4]));\nH = gpuArray(G, \"single\");\noriginal = gather(G);\nconverted = gather(H)",
        display_output: Some("original remains uint16; converted is single"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(original, \"uint16\"));\nassert(isa(converted, \"single\"));\nassert(isequal(original, uint16([1 2; 3 4])));\nassert(isequal(converted, single([1 2; 3 4])));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What can `gpuArray` upload?", answer: "The MATLAB-compatible form accepts real or complex numeric arrays and logical arrays. RunMat mode additionally supports character and string values where the active provider can represent the resulting storage." },
    BuiltinDocumentationFaq { question: "What happens when the input is already a gpuArray?", answer: "`gpuArray(G)` returns the existing resident value unchanged. A requested class conversion creates a new buffer and leaves `G` valid." },
    BuiltinDocumentationFaq { question: "Do class selectors, size arguments, and `like` work in MATLAB compatibility mode?", answer: "No. Those `gpuArray` constructor forms are RunMat extensions. Use `gpuArray(X)` with an explicitly converted or reshaped `X` for portable source." },
    BuiltinDocumentationFaq { question: "Does upload preserve fixed-width integers?", answer: "Yes. All eight fixed-width integer classes use native typed storage and retain their exact values and shape when the provider supports that physical element type." },
    BuiltinDocumentationFaq { question: "Does `gpuArray` choose a GPU automatically?", answer: "It uses the active acceleration provider and its selected device. If no provider is registered, the call returns a `gpuArray` provider error." },
    BuiltinDocumentationFaq { question: "When should I call `gather`?", answer: "Gather when host code or an unsupported operation needs the data. Keeping intermediate values resident avoids unnecessary transfers." },
    BuiltinDocumentationFaq { question: "Can a requested reshape change the number of elements?", answer: "No. RunMat's reshape-at-upload extension requires the requested dimensions to contain exactly the input element count." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
    BuiltinDocumentationLink {
        label: "gpuDevice",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuDevice"),
    },
    BuiltinDocumentationLink {
        label: "gpuInfo",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuInfo"),
    },
    BuiltinDocumentationLink {
        label: "arrayfun",
        target: BuiltinDocumentationLinkTarget::Builtin("arrayfun"),
    },
    BuiltinDocumentationLink {
        label: "zeros",
        target: BuiltinDocumentationLinkTarget::Builtin("zeros"),
    },
    BuiltinDocumentationLink {
        label: "sum",
        target: BuiltinDocumentationLinkTarget::Builtin("sum"),
    },
    BuiltinDocumentationLink {
        label: "pagefun",
        target: BuiltinDocumentationLinkTarget::Builtin("pagefun"),
    },
];

pub(in super::super) const GPUARRAY_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gpuArray"),
    slug: Some("gpuarray"),
    summary: "Move numeric or logical data to an acceleration provider.",
    description: "`gpuArray(X)` uploads data to the active provider and returns an explicitly resident value. RunMat mode also provides class selection, reshape-at-upload, and prototype-based construction.",
    keywords: &["gpuArray", "gpu", "device", "upload", "accelerate", "dtype", "like", "size"],
    related: &["arrayfun", "gather", "gpuDevice", "gpuInfo", "pagefun", "sum", "zeros"],
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
