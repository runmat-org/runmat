use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Log-gamma values and domain",
        paragraphs: &[
            "`gammaln(A)` evaluates the natural logarithm of the gamma function element by element for real, nonnegative input. Computing the logarithm directly avoids the overflow or underflow that can occur in `log(gamma(A))`.",
            "Zero and positive infinity return positive infinity, and NaN propagates. A negative real value produces a domain error rather than a complex result.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes, shapes, and RunMat extensions",
        paragraphs: &[
            "Documented input is dense real single or double. Output preserves the input floating class and shape. Complex, string, object, and sparse inputs are rejected.",
            "In RunMat compatibility mode, all eight fixed-width integer classes, logical values, and character arrays are also accepted. These extensions return double. Integer values must be exactly representable as binary64 at the calculation boundary; wide values that would lose bits are rejected before conversion.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated and distributed execution",
        paragraphs: &[
            "For real floating provider input, RunMat first proves the nonnegative domain with the input's exact owner. It validates a native `unary_gammaln` result's shape, real storage, precision, owner, device, class metadata, and non-aliasing, then restores the input's residency intent.",
            "A typed unsupported proof or unary hook enters exact-owner gather, host evaluation, and protected restoration. Other provider failures and malformed outputs remain visible. Integer and logical provider inputs use the authoritative host path before returning resident double output. Distributed arrays use partition-local unary mapping.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "scalar", title: "Evaluate a scalar", program: "Y = gammaln(5)", display_output: Some("Y = 3.1781"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(Y - log(24)) < 1e-12);" } },
    BuiltinExample { id: "half", title: "Evaluate a half-integer", program: "Y = gammaln(0.5)", display_output: Some("Y = 0.5724"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(Y - log(sqrt(pi))) < 1e-12);" } },
    BuiltinExample { id: "array", title: "Evaluate an array element by element", program: "A = [0.5 1 2 5];\nY = gammaln(A)", display_output: Some("Y = [0.5724 0 0 3.1781]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(Y - [log(sqrt(pi)) 0 0 log(24)])) < 1e-12);" } },
    BuiltinExample { id: "large", title: "Avoid overflow for a large input", program: "Y = gammaln(171)", display_output: Some("Y = 706.5731"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isfinite(Y));\nassert(abs(Y - 706.5730622457875) < 1e-10);" } },
    BuiltinExample { id: "single", title: "Preserve single precision", program: "Y = gammaln(single([0.5 1 5]))", display_output: Some("Y is a single row vector"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Y, 'single'));\nassert(max(abs(double(Y) - [log(sqrt(pi)) 0 log(24)])) < 2e-6);" } },
    BuiltinExample { id: "integer-extension", title: "Use exact integer input in RunMat mode", program: "Y = gammaln(uint16([1 5]))", display_output: Some("Y = [0 3.1781]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Y, 'double'));\nassert(max(abs(Y - [0 log(24)])) < 1e-12);" } },
    BuiltinExample { id: "gpu-residency", title: "Evaluate resident input", program: "G = gpuArray([0.5 1 5]);\nGy = gammaln(G);\nY = gather(Gy)", display_output: Some("Y = [0.5724 0 3.1781]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Gy, 'gpuArray'));\nassert(max(abs(Y - [log(sqrt(pi)) 0 log(24)])) < 1e-6);" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why use gammaln instead of log(gamma(A))?", answer: "`gammaln` computes the logarithm directly, so it can remain finite when forming `gamma(A)` first would overflow or underflow." },
    BuiltinDocumentationFaq { question: "Can A be negative or complex?", answer: "No. Numeric `gammaln` accepts real nonnegative input. Negative values produce a domain error, and complex values are rejected." },
    BuiltinDocumentationFaq { question: "What happens at zero?", answer: "`gammaln(0)` returns positive infinity." },
    BuiltinDocumentationFaq { question: "Which class does the result use?", answer: "Single input returns single and double input returns double. RunMat's integer, logical, and character extensions return double." },
    BuiltinDocumentationFaq { question: "Which integer values are accepted?", answer: "RunMat mode accepts all eight fixed-width classes when every value can cross the binary64 calculation boundary exactly. Values that would lose integer bits are rejected." },
    BuiltinDocumentationFaq { question: "Can provider input remain resident?", answer: "Yes. Valid native floating output remains resident; typed unsupported hooks and extension inputs use exact-owner gather and protected restoration." },
];

const RELATED: &[&str] = &["gamma", "factorial", "log", "gpuArray", "gather"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "gamma", target: BuiltinDocumentationLinkTarget::Builtin("gamma") },
    BuiltinDocumentationLink { label: "factorial", target: BuiltinDocumentationLinkTarget::Builtin("factorial") },
    BuiltinDocumentationLink { label: "log", target: BuiltinDocumentationLinkTarget::Builtin("log") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/gamma_functions/gammaln/mod.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Real log-gamma runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/gamma_functions/gammaln/mod.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Values, domain, classes, shapes, extensions, and rejection boundaries", location: "crates/runmat-runtime/src/builtins/math/elementwise/gamma_functions/gammaln/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Exact-owner fallback, resident double restoration, and explicit residency intent", location: "crates/runmat-runtime/src/builtins/math/elementwise/gamma_functions/gammaln/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU domain proof and direct-execution parity", location: "crates/runmat-runtime/src/builtins/math/elementwise/gamma_functions/gammaln/tests.rs" },
    ],
    notes: &[],
};

pub(super) const GAMMALN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gammaln"), slug: Some("gammaln"),
    summary: "Evaluate the natural logarithm of the gamma function for real nonnegative input.",
    description: "`gammaln` computes log-gamma values directly, preserves documented floating classes and shapes, and supports validated provider-resident execution.",
    keywords: &["gammaln", "gamma", "log gamma", "special function", "single", "double", "gpu"],
    related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[], evidence: EVIDENCE,
    introduced: Some("Before R2006a"), status: Some(BuiltinDocumentationStatus::Stable),
};
