use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("histcounts2"),
    slug: Some("histcounts2"),
    summary: "Count paired observations in 2-D histogram bins in MATLAB and RunMat.",
    description: "`histcounts2(X, Y)` bins paired observations into a rectangular two-dimensional histogram. Edge handling, bin controls, and output layout follow MATLAB semantics.",
    keywords: &[
        "histcounts2",
        "2d histogram",
        "joint distribution",
        "binning",
        "probability",
        "gpu"],
    related: &[
        "histcounts"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`histcounts2(X, Y)` requires X and Y to have the same shape, then pairs their column-major elements, and fills a `numel(Xedges) - 1` by `numel(Yedges) - 1` matrix of bin counts.",
                "Bins are half-open on the right except for the last column and row, which include their upper edges so the maximum finite values are counted.",
                "Optional arguments let you specify bin counts, explicit edges, bin limits, bin widths, and automatic binning heuristics independently for the `X` and `Y` axes.",
                "Name/value pairs such as `'NumBins'`, `'XBinEdges'`, `'YBinEdges'`, `'XBinWidth'`, `'YBinWidth'`, `'BinMethod'`, and `'Normalization'` follow MATLAB precedence and validation rules.",
                "Pairs containing `NaN` in either coordinate are ignored. Infinite values participate when the chosen edges include them."]
        },
        BuiltinDocumentationSection {
            heading: "Syntax and outputs",
            paragraphs: &[
                "[N, Xedges, Yedges] = histcounts2(X, Y)\n[N, Xedges, Yedges] = histcounts2(X, Y, nbins)\n[N, Xedges, Yedges] = histcounts2(X, Y, Xedges, Yedges)\n[N, Xedges, Yedges] = histcounts2(___, Name, Value)\n[N, Xedges, Yedges, binX, binY] = histcounts2(___)",
                "`X` and `Y` are numeric vectors (or arrays flattened column-major) of equal length holding paired observations.",
                "`nbins` is either a scalar bin count applied to both axes or a two-element vector `[nX nY]` giving separate counts for the `X` and `Y` axes.",
                "`Xedges` and `Yedges` are monotonically increasing edge vectors. Bin `k` covers `[edge(k), edge(k+1))`; the final bin also includes its right edge so maximum values are counted.",
                "Name/value options include `'Normalization'` (`'count'`, `'probability'`, `'pdf'`, `'countdensity'`, `'cumcount'`, `'cdf'`), `'BinMethod'`, `'BinWidth'`, `'NumBins'`, `'XBinLimits'`, and `'YBinLimits'`.",
                "With five outputs, `binX` and `binY` are integer indices the same shape as the corresponding inputs, giving the `X`- and `Y`-bin that each pair falls into (`0` when an observation is excluded)."]
        },
        BuiltinDocumentationSection {
            heading: "Classes and execution",
            paragraphs: &[
                "Counts, edges, and bin-index outputs are host double arrays. For explicit integer edges, fixed-width integer observations are compared exactly, including values beyond flintmax; generated and returned edges are double. Integer-aligned unit bins require distinct double boundaries on each axis; RunMat reports an edge-precision error when they cannot be represented.",
                "Resident inputs are gathered through their providers before the CPU histogram runs. This materializing operation has no implemented provider reduction or fusion kernel."]
        }
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Counting paired values with explicit edges",
            program: "X = [0.5 1.5 2.5 3.5];\nY = [0.2 0.9 1.4 2.8];\n[N, Xedges, Yedges] = histcounts2(X, Y, [0 1 2 3 4], [0 1 2 3])",
            display_output: Some("N = [1 0 0; 1 0 0; 0 1 0; 0 0 1];\nXedges = [0 1 2 3 4];\nYedges = [0 1 2 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(N,[1 0 0;1 0 0;0 1 0;0 0 1])); assert(isequal(Xedges,[0 1 2 3 4])); assert(isequal(Yedges,[0 1 2 3]));" }
        },
        BuiltinExample {
            id: "example-2",
            title: "Specifying separate bin counts for each axis",
            program: "X = [1 2 3 4];\nY = [1 2 3 4];\nN = histcounts2(X, Y, 2, 4);\nsize(N)\nsum(N, \"all\")",
            display_output: Some("size(N) = [2 4];\nsum(N, \"all\") = 4"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(N),[2 4])); assert(sum(N(:))==4);" }
        },
        BuiltinExample {
            id: "example-3",
            title: "Using different bin widths for X and Y",
            program: "X = [1 1.5 2.4 3.7];\nY = [2 2.2 2.9 3.1];\n[N, Xedges, Yedges] = histcounts2(X, Y, 'XBinWidth', 1, 'YBinWidth', 0.5);\ndiff(Xedges)\ndiff(Yedges(1:3))\nsum(N, \"all\")",
            display_output: Some("diff(Xedges) = [1 1 1];\ndiff(Yedges(1:3)) = [0.5 0.5];\nsum(N, \"all\") = 4"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(sum(N(:))==4); assert(all(abs(diff(Xedges)-1)<1e-12)); assert(all(abs(diff(Yedges(1:3))-0.5)<1e-12));" }
        },
        BuiltinExample {
            id: "example-4",
            title: "Normalizing a 2-D histogram to probabilities",
            program: "X = [0.2 0.4 1.1 1.5];\nY = [0.1 0.8 1.2 1.9];\nN = histcounts2(X, Y, [0 1 2], [0 1 2], 'Normalization', 'probability')",
            display_output: Some("N = [0.5 0.0; 0.0 0.5]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(N,[0.5 0;0 0.5]));" }
        },
        BuiltinExample {
            id: "example-5",
            title: "Ignoring NaN values in paired data",
            program: "X = [1 2 NaN 3];\nY = [2 2 2 NaN];\nN = histcounts2(X, Y, [0 1 2 3], [0 1 2 3]);\nsum(N, \"all\")",
            display_output: Some("sum(N, \"all\") = 2"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(sum(N(:))==2);" }
        },
        BuiltinExample {
            id: "example-6",
            title: "Histogramming gpuArray inputs without manual gather",
            program: "Gx = gpuArray([0.5 1.5 2.5]);\nGy = gpuArray([1.0 1.1 2.9]);\n[counts, Xedges, Yedges] = histcounts2(Gx, Gy, [0 1 2 3], [0 2 3]);\nisa(counts, 'double')      % counts are returned on the CPU\ncounts",
            display_output: Some("isa(counts, 'double')      % counts are returned on the CPU\ncounts = [1 0; 1 0; 0 1]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(counts,[1 0;1 0;0 1])); assert(isequal(Xedges,[0 1 2 3])); assert(isequal(Yedges,[0 2 3]));" }
        }
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Do `X` and `Y` need the same size?",
            answer: "Yes. `histcounts2` requires both inputs to have the same shape. RunMat raises an error when the shapes differ, even if the element counts match." },
        BuiltinDocumentationFaq {
            question: "How are the bin edges interpreted?",
            answer: "All interior bins are `[left, right)`, while the last row and column are `[left, right]`, so the largest finite values are counted." },
        BuiltinDocumentationFaq {
            question: "What happens to `NaN`, `Inf`, or `-Inf` values?",
            answer: "Pairs containing `NaN` in either coordinate are ignored. Infinite values participate when the specified edges include them; otherwise, they are excluded just like MATLAB." },
        BuiltinDocumentationFaq {
            question: "Can I mix explicit edges on one axis with automatic binning on the other?",
            answer: "Yes. You can supply `'XBinEdges'` while leaving the `Y` axis to be determined by `'NumBins'`, `'YBinWidth'`, or the default heuristics." },
        BuiltinDocumentationFaq {
            question: "Which normalisation modes are supported?",
            answer: "`'count'`, `'probability'`, `'countdensity'`, `'pdf'`, `'cumcount'`, and `'cdf'` are implemented. `'cdf'` and `'cumcount'` operate in column-major order so the result matches MATLAB's cumulative behaviour." },
        BuiltinDocumentationFaq {
            question: "How do I request integer-aligned bins?",
            answer: "Use `'BinMethod', 'integers'` or `'XBinMethod'/'YBinMethod'` with the value `'integers'`. RunMat ensures the resulting edges align with integer boundaries, respecting any supplied bin limits." },
        BuiltinDocumentationFaq {
            question: "What does histcounts2 do in MATLAB?",
            answer: "`histcounts2` bins paired observations into a two-dimensional histogram. It accepts two vectors `X` and `Y` of equal length and returns a matrix of bin counts plus the bin edges for each axis." },
        BuiltinDocumentationFaq {
            question: "How is histcounts2 different from histcounts?",
            answer: "`histcounts` bins data along a single dimension, while `histcounts2` bins paired data along two dimensions simultaneously, producing a 2-D count matrix." },
        BuiltinDocumentationFaq {
            question: "Can I use histcounts2 on GPU arrays in RunMat?",
            answer: "Yes. RunMat supports `histcounts2` on GPU arrays with `f32` and `f64` precision. The current implementation gathers to host; no GPU histogram kernel is registered." },
        BuiltinDocumentationFaq {
            question: "What's the relationship between `N`, `Xedges`, and `Yedges`?",
            answer: "The count matrix `N` has size `(length(Xedges)-1) × (length(Yedges)-1)`: one row per `X` bin and one column per `Y` bin. Each bin covers the half-open interval `[edge(k), edge(k+1))`, except the final row and column, which include their upper edge so the largest finite values are counted." },
        BuiltinDocumentationFaq {
            question: "How do I get a 2-D probability density from `histcounts2`?",
            answer: "Pass `'Normalization', 'pdf'`, for example `[N, Xedges, Yedges] = histcounts2(X, Y, 'Normalization', 'pdf')`. Each bin value is divided by the bin area and by the total number of pairs, so `sum(N .* diff(Xedges)' .* diff(Yedges), 'all')` equals `1` when no samples fall outside the bin limits." },
        BuiltinDocumentationFaq {
            question: "How do I use explicit bin edges with `histcounts2`?",
            answer: "Supply two monotonically increasing edge vectors as the third and fourth arguments, for example `Xedges = 0:0.1:1; Yedges = 0:0.1:1; N = histcounts2(X, Y, Xedges, Yedges)`. This gives you full control over bin placement and is the easiest way to reproduce a 2-D histogram across multiple datasets." }
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "histcounts",
            target: BuiltinDocumentationLinkTarget::Builtin("histcounts") },
        BuiltinDocumentationLink {
            label: "sum",
            target: BuiltinDocumentationLinkTarget::Builtin("sum") },
        BuiltinDocumentationLink {
            label: "gpuArray",
            target: BuiltinDocumentationLinkTarget::Builtin("gpuarray") }
    ],
    media: &[
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Source",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/hist/histcounts2.rs") }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Histogram semantics",
                location: "crates/runmat-runtime/src/builtins/stats/hist/histcounts2/tests.rs" },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Resident provider round trip",
                location: "crates/runmat-runtime/src/builtins/stats/hist/histcounts2/tests.rs#histcounts2_gpu_roundtrip" },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::WgpuTest,
                label: "WGPU histogram round trip",
                location: "crates/runmat-runtime/src/builtins/stats/hist/histcounts2/tests.rs#histcounts2_wgpu_roundtrip" }
        ],
        notes: &[
        ]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
