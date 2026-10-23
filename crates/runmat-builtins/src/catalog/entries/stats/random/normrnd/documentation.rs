//! Conserved reference prose and independently asserted examples.
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`normrnd(mu, sigma)` returns a scalar drawn from `N(mu, sigma^2)`.",
            "`normrnd(mu, sigma, n)` returns an `n × n` matrix of samples.",
            "`normrnd(mu, sigma, m, n)` returns an `m × n` matrix of samples.",
            "`normrnd(mu, sigma, sz)` accepts a size vector and returns an array with shape `sz`.",
            "`mu` and `sigma` must be scalar numeric values.",
            "Current MATLAB documents single and double mu, sigma, and size values. Typed-integer and logical inputs are independently gated RunMat extensions; integer parameters must be exactly representable at the binary64 sampling boundary.",
            "Integer size controls are parsed structurally and do not select output class or residency. Explicit resident size controls are a separate RunMat extension.",
            "`sigma` must be non-negative; `normrnd` errors if `sigma < 0`.",
            "Any single mu or sigma selects single output; otherwise output is double. A resident data parameter selects the exact owner for class-preserving restoration. Unsupported automatic restoration can keep the host output; explicit gpuArray intent requires restoration.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution and residency",
        paragraphs: &["Sampling currently uses the shared host normal generator, then restores a resident data parameter result through its exact owning provider. No random_normrnd provider hook is invoked by this implementation. Restoration validates the requested single/double storage class, shape, owner and non-aliasing; typed pre-effect unsupported automatic restoration may keep the host result, while explicit gpuArray intent requires successful restoration. Host parameters do not opportunistically select a global provider."],
    },
    BuiltinDocumentationSection {
        heading: "Random state and fusion",
        paragraphs: &["Random generation and RNG state control are stateful boundaries and are not fused into elementwise kernels."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: f32. See the distribution, integer-extension, and residency rules above."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: f64. See the distribution, integer-extension, and residency rules above."],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation limits",
        paragraphs: &["The implemented distribution parameters mu and sigma are scalar. Array-valued distribution parameters are not implemented."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Single sample from N(0, 1)",
        program: "rng(0);\nr = normrnd(0, 1)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isscalar(r)); assert(isa(r,'double')); assert(isfinite(r)); rng(0); assert(isequal(r,normrnd(0,1)));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Matrix of Gaussian noise with mean 10 and standard deviation 2",
        program: "rng(0);\nX = normrnd(10, 2, 2, 3)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(X),[2 3])); assert(all(isfinite(X(:)))); rng(0); assert(isequal(X,normrnd(10,2,2,3)));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Simulating normally-distributed measurement errors",
        program: "rng(0);\ntrueValue = 100;\nsigma = 0.5;\nmeasurements = trueValue + normrnd(0, sigma, 1, 5)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(measurements),[1 5])); assert(all(isfinite(measurements)));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Specifying dimensions with a size vector",
        program: "rng(0);\nT = normrnd(5, 1.5, [2 3])",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(T),[2 3])); assert(all(isfinite(T(:))));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Select single output from a single data parameter",
        program: "rng(0); R = normrnd(single(2),single(0),2,3);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(R,'single')); assert(isequal(size(R),[2 3])); assert(all(R(:)==single(2)));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("normrnd"),
    slug: Some("normrnd"),
    summary: "Generate normally distributed random numbers in MATLAB and RunMat.",
    description: "`normrnd(mu, sigma, ...)` draws pseudorandom samples from a normal distribution with scalar mean `mu` and standard deviation `sigma`. Scalar and vector size forms control the output shape.",
    keywords: &["normrnd", "normal", "gaussian", "random", "distribution", "statistics"],
    related: &[],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What do mu and sigma mean?",
            answer: "`mu` is the distribution mean and `sigma` is the standard deviation. The variance is `sigma^2`, so `normrnd(mu, sigma)` draws from `N(mu, sigma^2)`.",
        },
        BuiltinDocumentationFaq {
            question: "What is the workaround if normrnd is unavailable?",
            answer: "`normrnd(mu, sigma)` is mathematically equivalent to `mu + sigma * randn()`. For arrays, use `mu + sigma * randn(sz)` with the same requested size.",
        },
        BuiltinDocumentationFaq {
            question: "Can sigma be zero?",
            answer: "Yes. `sigma = 0` is allowed and returns deterministic samples equal to `mu`. Negative standard deviations are undefined and raise an error.",
        },
        BuiltinDocumentationFaq {
            question: "What use cases does normrnd support?",
            answer: "Common uses include Gaussian noise generation, measurement error simulation, confidence interval experiments, Monte Carlo sampling, and normally-distributed model residuals.",
        },
        BuiltinDocumentationFaq {
            question: "Does normrnd fuse with other operations?",
            answer: "No. Random generation is excluded from fusion planning to preserve statistical properties.",
        },
        BuiltinDocumentationFaq {
            question: "How do I control reproducibility?",
            answer: "Use `rng` before calling `normrnd` to seed the global generator.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "exprnd",
            target: BuiltinDocumentationLinkTarget::Builtin("exprnd"),
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
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/stats/random/normrnd.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/normrnd.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::stats::random::normrnd::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration",
                location: "crates/runmat-vm/tests/integer_normal_distribution_and_regression_semantics.rs",
            },
        ],
        notes: &["Integer compatibility and placement are declared in the canonical contract. Examples assert shape, class, bounds, or state replay rather than fixed random transcripts."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
