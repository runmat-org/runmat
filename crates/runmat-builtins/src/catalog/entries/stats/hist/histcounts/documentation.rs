use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("histcounts"),
    slug: Some("histcounts"),
    summary: "Count observations in histogram bins in MATLAB and RunMat.",
    description: "`histcounts` tallies how many values fall in each histogram bin. Its optional third output gives one-based bin indices in the input shape, with zero for excluded values. Explicit edges, bin-count/bin-width controls, and default bin selection follow MATLAB semantics.",
    keywords: &[
        "histcounts",
        "histogram",
        "binning",
        "normalization",
        "probability",
        "cdf",
        "gpu"],
    related: &[
        "histcounts2"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`histcounts(X)` flattens numeric or logical inputs column-major and returns a row vector of counts spread across ten equal-width bins spanning the data range.",
                "`histcounts(X, N)` partitions the data into `N` equally spaced bins.",
                "`histcounts(X, edges)` counts observations using the supplied bin edges.",
                "Name/value pairs such as `'BinWidth'`, `'BinLimits'`, `'NumBins'`, `'BinEdges'`, `'BinMethod'`, and `'Normalization'` follow MATLAB's precedence rules and validation logic.",
                "Values outside the bin limits are excluded. The last bin includes its upper edge while all other bins are half-open on the right.",
                "`NaN` values are ignored; `Inf` and `-Inf` participate when the edges cover them."]
        },
        BuiltinDocumentationSection {
            heading: "Classes and execution",
            paragraphs: &[
                "Counts and edges are host double arrays. For explicit integer edges, fixed-width integer observations are compared exactly, including values beyond flintmax; generated and returned edges are double. Integer-aligned unit bins require distinct double boundaries; RunMat reports an edge-precision error when they cannot be represented.",
                "Resident inputs are gathered through the owning provider before the CPU histogram runs. This materializing operation has no implemented provider reduction or fusion kernel. 64-bit integer gpuArray inputs are rejected before gathering."]
        }
    ],
    examples: &[
        BuiltinExample {
            id: "bin-indices",
            title: "Locate each observation's bin",
            program: "data = [0 NaN 1 2 3];\n[counts, edges, bin] = histcounts(data, [0 1 2 3])",
            display_output: Some("counts = [1 1 2];\nedges = [0 1 2 3];\nbin = [1 0 2 3 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(counts,[1 1 2])); assert(isequal(edges,[0 1 2 3])); assert(isequal(bin,[1 0 2 3 3]));" }
        },
        BuiltinExample {
            id: "example-1",
            title: "Counting values with custom bin counts",
            program: "data = [1 2 2 4 5 7];\n[counts, edges] = histcounts(data, 3)",
            display_output: Some("counts = [3 1 2];\nedges  = [1 3 5 7]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(counts,[3 1 2])); assert(isequal(edges,[1 3 5 7]));" }
        },
        BuiltinExample {
            id: "example-2",
            title: "Using explicit bin edges",
            program: "edges = [0 1 2 3];\ncounts = histcounts([0.1 0.5 0.9 1.2 1.8 2.1], edges)",
            display_output: Some("counts = [3 2 1]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(counts,[3 2 1]));" }
        },
        BuiltinExample {
            id: "example-3",
            title: "Setting bin width and limits",
            program: "[counts, edges] = histcounts([5 7 8 10 12], 'BinWidth', 2, 'BinLimits', [4 12])",
            display_output: Some("counts = [1 1 1 2];\nedges  = [4 6 8 10 12]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(counts,[1 1 1 2])); assert(isequal(edges,[4 6 8 10 12]));" }
        },
        BuiltinExample {
            id: "example-4",
            title: "Choosing an automatic binning method",
            program: "[counts, edges] = histcounts(randn(1, 500), 'BinMethod', 'sturges');\nnumel(counts)",
            display_output: Some("numel(counts) = ceil(log2(500) + 1);   % 10 bins"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(counts)==10); assert(sum(counts)==500);" }
        },
        BuiltinExample {
            id: "example-5",
            title: "Normalising counts to probabilities",
            program: "counts = histcounts([0.2 0.4 1.1 1.4 1.8 2.5], [0 1 2 3], 'Normalization', 'probability')",
            display_output: Some("counts = [0.3333 0.5000 0.1667]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(all(abs(counts-[1/3 1/2 1/6])<1e-12));" }
        },
        BuiltinExample {
            id: "example-6",
            title: "Building a cumulative distribution",
            program: "counts = histcounts([1 2 2 3], [0 1 2 3], 'Normalization', 'cdf')",
            display_output: Some("counts = [0 0.25 1]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(counts,[0 0.25 1]));" }
        },
        BuiltinExample {
            id: "example-7",
            title: "Counting values stored on a GPU array",
            program: "G = gpuArray([0.5 1.5 2.5]);\n[counts, edges] = histcounts(G, [0 1 2 3])   % counts/edges return as CPU arrays",
            display_output: Some("counts = [1 1 1];\nedges  = [0 1 2 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(counts,[1 1 1])); assert(isequal(edges,[0 1 2 3]));" }
        }
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Why does the last bin include its upper edge?",
            answer: "To match MATLAB semantics each bin is `[left, right)` except for the final bin, which is `[left, right]`. This ensures the maximum finite value is always counted." },
        BuiltinDocumentationFaq {
            question: "How are `NaN` values handled?",
            answer: "They are ignored entirely and do not contribute to any bin count. Infinite values participate as long as the bin edges include them." },
        BuiltinDocumentationFaq {
            question: "What happens when all observations are identical?",
            answer: "RunMat mirrors MATLAB by collapsing the histogram to a single bin centred on the shared value unless you explicitly supply edges, limits, or a bin width." },
        BuiltinDocumentationFaq {
            question: "Does `histcounts` support non-double inputs?",
            answer: "Yes. Logical inputs are promoted to doubles, integer observations retain their exact values for bin comparisons, and gpuArray inputs are gathered to host memory in this release." },
        BuiltinDocumentationFaq {
            question: "Can I request both `'BinEdges'` and `'BinWidth'`?",
            answer: "No. Bin specifications are mutually exclusive—choose one of `'BinEdges'`, `'BinWidth'`, or `'NumBins'`, optionally constrained by `'BinLimits'`." },
        BuiltinDocumentationFaq {
            question: "How do probability and PDF normalisations differ?",
            answer: "`'probability'` scales counts so that they sum to one. `'pdf'` divides by both bin width and the total count, matching MATLAB's probability-density definition." },
        BuiltinDocumentationFaq {
            question: "Do outputs stay on the GPU when the input is a `gpuArray`?",
            answer: "RunMat gathers GPU data to the CPU and returns host-resident outputs. Use `gather` only for clarity; the values are already in host memory." }
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "linspace",
            target: BuiltinDocumentationLinkTarget::Builtin("linspace") },
        BuiltinDocumentationLink {
            label: "sum",
            target: BuiltinDocumentationLinkTarget::Builtin("sum") },
        BuiltinDocumentationLink {
            label: "mean",
            target: BuiltinDocumentationLinkTarget::Builtin("mean") },
        BuiltinDocumentationLink {
            label: "rand",
            target: BuiltinDocumentationLinkTarget::Builtin("rand") },
        BuiltinDocumentationLink {
            label: "histcounts2",
            target: BuiltinDocumentationLinkTarget::Builtin("histcounts2") }
    ],
    media: &[
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Source",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/hist/histcounts.rs") }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Histogram semantics",
                location: "crates/runmat-runtime/src/builtins/stats/hist/histcounts/tests.rs" },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Resident provider round trip",
                location: "crates/runmat-runtime/src/builtins/stats/hist/histcounts/tests.rs#histcounts_gpu_roundtrip" },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::WgpuTest,
                label: "WGPU histogram round trip",
                location: "crates/runmat-runtime/src/builtins/stats/hist/histcounts/tests.rs#histcounts_wgpu_roundtrip" }
        ],
        notes: &[
        ]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
