use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Represented spacing",
        paragraphs: &[
            "Bare `eps` and `eps()` return the double-precision gap above one, exactly 2^-52. `eps(\"single\")` returns the single-precision gap above one, exactly 2^-23.",
            "The bare constant is a host double scalar obtained without array allocation, provider access, ambient services, or effects. Constant lookup has its own typed binding; it does not invoke the callable forms. Native and WASM consumers use the same constant value.",
            "For a real or complex floating array `x`, `eps(x)` has the same shape and floating class as `x` and contains the positive spacing above each element's magnitude. The result is real and dense, including for complex or sparse input (implicit sparse zeros become positive subnormal spacings). Zero and subnormal magnitudes use the smallest positive subnormal spacing; Inf and NaN produce NaN. Integer and logical arrays are not accepted as value inputs.",
            "`eps(\"like\", p)` returns the spacing at one as a scalar matching the floating class, complexity, sparsity, and supported residency of `p`. It does not inspect the prototype's numeric contents.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resident values",
        paragraphs: &[
            "A supported real GPU input is evaluated on its owning provider and yields a resident output with the same shape and precision. The `like` form creates a resident scalar through protected provider upload. Unsupported resident classes or provider operations are rejected without silently downloading an explicitly resident input.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "value",
        title: "Read the constant",
        program: "value = eps",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(value, 'double')); assert(isscalar(value)); assert(value == 2^-52); assert(1 + value > 1);",
        },
    },
    BuiltinExample {
        id: "spacing-at-one",
        title: "Compare double and single spacing",
        program: "d = eps; s = eps('single'); disp(d); disp(s)",
        display_output: Some("d = 2.2204e-16\ns = single(1.1921e-07)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(d == 2^-52); assert(isa(s,'single')); assert(s == single(2^-23));",
        },
    },
    BuiltinExample {
        id: "array-binades",
        title: "Measure spacing across binades",
        program: "x = [0 1 2 -2]; d = eps(x); disp(d)",
        display_output: Some("d is a 1-by-4 double row vector of positive spacings"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(d),size(x))); assert(d(1) > 0); assert(d(2) == 2^-52); assert(d(3) == 2^-51); assert(d(4) == d(3));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("eps"),
    slug: Some("eps"),
    summary: "Return floating-point spacing at one or at an input magnitude.",
    description: "`eps` is a double constant; `eps()`, `eps(x)`, `eps(typename)`, and `eps(\"like\", p)` are callable forms. The array form computes exact represented spacing rather than multiplying the magnitude by machine epsilon.",
    keywords: &["eps", "spacing", "single", "double", "subnormal", "gpu"],
    related: &["realmin", "realmax", "flintmax"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What does `eps(0)` return?",
            answer: "The smallest positive subnormal number of the input's floating class, not zero or the smallest normalized number.",
        },
        BuiltinDocumentationFaq {
            question: "Does the bare constant select GPU residency?",
            answer: "No. Bare eps is a host scalar; ordinary numeric operations may subsequently consume or transfer it.",
        },
    ],
    links: &[],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime spacing implementation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/numeric_limits/eps.rs"),
            },
            BuiltinDocumentationLink {
                label: "Bare constant binding",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/constants/core/eps.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Numeric limit spacing tests",
            location: "crates/runmat-runtime/src/builtins/math/elementwise/numeric_limits/tests/floating.rs",
        }],
        notes: &["The callable and bare-constant spellings have one catalog documentation authority."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
