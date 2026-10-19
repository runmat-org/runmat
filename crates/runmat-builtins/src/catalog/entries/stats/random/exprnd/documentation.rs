use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleFixture,
    BuiltinExampleHarness, BuiltinExampleRequirements, BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`exprnd(mu)` returns one sample per element of `mu` and preserves the shape and single/double precision of `mu`.",
            "`exprnd(mu, n)` returns an `n × n` matrix of samples.",
            "`exprnd(mu, m, n)` returns an `m × n` matrix of samples.",
            "`exprnd(mu, sz)` accepts a size vector and returns an array with shape `sz`.",
            "A scalar `mu` expands to an explicitly requested output shape; a nonscalar `mu` requires that shape to match exactly.",
            "Zero or negative size extents produce an empty array, and trailing singleton dimensions beyond dimension two are ignored.",
            "`mu` must contain finite positive values; logical and complex means reject.",
            "MATLAB documents `mu` and size controls as single or double. RunMat can additionally accept typed-integer means and sizes only when RunMat extensions are enabled.",
            "Resident fallback returns output only to the exact mean input owner and only when that owner physically supports the required precision. Automatic placement may remain on the host; explicit gpuArray input requires restoration and otherwise fails before sampling. Native array-mean/single provider hooks and independent CPU/GPU random streams remain provider-ABI work.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution and fusion",
        paragraphs: &[
            "The scalar-double provider hook is used only when its returned handle passes ownership, shape, storage, class, and physical-precision validation before RNG state advances. Array means, native single, and resident means use host fallback; restoration requires the exact source owner to physically support the output precision.",
            "Random generation is a stateful, non-fusible operation. Successful sampling advances the shared RNG stream even when the caller discards the result; invalid arguments and excess output requests fail before sampling.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "single-array",
        title: "Sample from single-precision means",
        program: "rng(0); mu = single([1 2 3]); r = exprnd(mu);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(r, 'single')); assert(isequal(size(r), [1 3])); assert(all(r > 0)); rng(0); assert(isequal(r, exprnd(mu)));",
        },
    },
    BuiltinExample {
        id: "integer-extension",
        title: "Use fixed-width means and dimensions",
        program: "rng(0); r = exprnd(uint16(2), uint8([2 3]));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(r, 'double')); assert(isequal(size(r), [2 3])); assert(all(r(:) > 0)); rng(0); assert(isequal(r, exprnd(2, [2 3])));",
        },
    },
    BuiltinExample {
        id: "resident-single",
        title: "Retain an explicit GPU result",
        program: "rng(0); mu = gpuArray(single([1 2 3])); gr = exprnd(mu); r = gather(gr);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isgpuarray(gr)); assert(isa(r, 'single')); assert(isequal(size(r), [1 3])); assert(all(r > 0)); rng(0); expected = exprnd(single([1 2 3])); assert(isequal(r, expected));",
        },
    },
    BuiltinExample {
        id: "example-1",
        title: "Single sample from Exp(2)",
        program: "rng(0);\nr = exprnd(2)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isscalar(r)); assert(isa(r, 'double')); assert(r > 0); rng(0); again = exprnd(2); assert(isequal(r, again));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Matrix of arrival times for a Poisson process (lambda = 2)",
        program: "rng(0);\nlambda = 2;\nintervals = exprnd(1/lambda, 1, 5)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(intervals), [1 5])); assert(all(intervals > 0)); rng(0); again = exprnd(1/lambda, 1, 5); assert(isequal(intervals, again));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "M/M/1 queueing simulation fragment",
        program: "rng(0);\nlambda = 2; mu_svc = 2.5; Tsim = 100;\nt = 0; arrivals = 0;\nwhile t < Tsim\n    t = t + exprnd(1/lambda);\n    arrivals = arrivals + 1;\nend\narrivals",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(t >= Tsim); assert(arrivals > 0); assert(arrivals == floor(arrivals));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Specifying dimensions with a size vector",
        program: "rng(0);\nT = exprnd(1, [2 3])",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(T), [2 3])); assert(all(T(:) > 0)); rng(0); again = exprnd(1, [2 3]); assert(isequal(T, again));",
        },
    },
];
pub(super) const EXPRND_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("exprnd"),
    slug: Some("exprnd"),
    summary: "Generate exponentially distributed random samples with mean `mu` and MATLAB-compatible size handling.",
    description: "`exprnd(mu, ...)` draws exponential random samples using mean parameter `mu`. It supports MATLAB-compatible scalar expansion and explicit output-size signatures.",
    keywords: &["exprnd", "exponential", "random", "distribution", "statistics", "queueing", "poisson"],
    related: &["normrnd", "unifrnd", "rand", "randn", "rng"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What is the relationship between mu and lambda?",
            answer: "The exponential distribution can be parameterised by its mean `mu` or its rate `lambda`. They are reciprocals: `mu = 1/lambda`. `exprnd(mu)` is equivalent to `exprnd(1/lambda)`.",
        },
        BuiltinDocumentationFaq {
            question: "What is the workaround if exprnd is unavailable?",
            answer: "`exprnd(mu)` is mathematically equivalent to `-mu * log(rand())`. You can substitute `-log(rand())/lambda` inline until `exprnd` is available.",
        },
        BuiltinDocumentationFaq {
            question: "Why must mu be positive?",
            answer: "A non-positive mean is undefined for the exponential distribution. `exprnd` raises an error if `mu <= 0` to catch common parameter mistakes early.",
        },
        BuiltinDocumentationFaq {
            question: "What use cases does exprnd support?",
            answer: "Common uses include queueing theory (M/M/1, M/G/1 inter-arrival and service times), reliability engineering (time-to-failure models), Poisson process simulation, and Monte Carlo sampling.",
        },
        BuiltinDocumentationFaq {
            question: "Does exprnd fuse with other operations?",
            answer: "No. Random generation is excluded from fusion planning to preserve statistical properties.",
        },
        BuiltinDocumentationFaq {
            question: "How do I control reproducibility?",
            answer: "Use `rng` before calling `exprnd` to seed the global generator.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "normrnd",
            target: BuiltinDocumentationLinkTarget::Builtin("normrnd"),
        },
        BuiltinDocumentationLink {
            label: "unifrnd",
            target: BuiltinDocumentationLinkTarget::Builtin("unifrnd"),
        },
        BuiltinDocumentationLink {
            label: "rand",
            target: BuiltinDocumentationLinkTarget::Builtin("rand"),
        },
        BuiltinDocumentationLink {
            label: "randn",
            target: BuiltinDocumentationLinkTarget::Builtin("randn"),
        },
        BuiltinDocumentationLink {
            label: "rng",
            target: BuiltinDocumentationLinkTarget::Builtin("rng"),
        },
        BuiltinDocumentationLink {
            label: "Implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/exprnd.rs"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Shape, precision, compatibility, state, and provider validation",
            location: "crates/runmat-runtime/src/builtins/stats/random/exprnd/tests.rs",
        }],
        notes: &["Legacy GPU and fusion fields are represented in the typed placement contract and GPU section; legacy tested/source links are retained as evidence and links."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
