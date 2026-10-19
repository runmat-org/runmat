const SECTIONS: &[crate::BuiltinDocumentationSection] = &[
    crate::BuiltinDocumentationSection { heading: "Truth and shape semantics", paragraphs: &["`not(A)` and `~A` invert each element's zero/nonzero truth value. Zero becomes true; every nonzero numeric value, including NaN and a complex value with either component nonzero, becomes false.", "All eight fixed-width integer classes are tested directly in authoritative storage. Array shape and empty geometry are preserved."] },
    crate::BuiltinDocumentationSection { heading: "Tables and accelerated execution", paragraphs: &["Tables and timetables apply `not` independently to every supported variable and retain their container metadata.", "A supported same-owner provider operation remains resident. Unsupported provider routes gather exactly; explicitly requested residency is restored only after validating owner, device, shape, storage, and aliasing."] },
];
const EXAMPLES: &[crate::BuiltinExample] = &[
    crate::BuiltinExample { id: "scalar", title: "Negate a scalar condition", program: "tf = not(5)", display_output: Some("tf = false"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, false));" } },
    crate::BuiltinExample { id: "array", title: "Invert a logical mask", program: "tf = not(logical([1 0 1]))", display_output: Some("tf = [false true false]"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([0 1 0])));" } },
    crate::BuiltinExample { id: "numeric", title: "Invert numeric truth values", program: "tf = not([0 1 NaN 0])", display_output: Some("tf = [true false false true]"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([1 0 0 1])));" } },
    crate::BuiltinExample { id: "integer", title: "Evaluate wide integers exactly", program: "x = [uint64(0) intmax('uint64')];\ntf = not(x)", display_output: Some("tf = [true false]"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([1 0])));" } },
    crate::BuiltinExample { id: "characters", title: "Invert character code points", program: "tf = not(['A' 0 'C'])", display_output: Some("tf = [false true false]"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([0 1 0])));" } },
    crate::BuiltinExample { id: "complex", title: "Invert complex truth values", program: "tf = not([0+0i 1+0i 0+2i])", display_output: Some("tf = [true false false]"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([1 0 0])));" } },
    crate::BuiltinExample { id: "table", title: "Invert table variables", program: "T = table([0; 2], [1; 0], 'VariableNames', {'A', 'B'});\nR = not(T)", display_output: Some("R is a table with logical variables"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(istable(R)); assert(isequal(R.A, logical([1; 0]))); assert(isequal(R.B, logical([0; 1])));" } },
    crate::BuiltinExample { id: "gpu", title: "Keep a supported result resident", program: "A = gpuArray([0 2 0]);\nG = not(A);\ntf = gather(G)", display_output: Some("G remains resident and tf = [true false true]"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isa(G, 'gpuArray')); assert(isequal(tf, logical([1 0 1])));" } },
];
const FAQS: &[crate::BuiltinDocumentationFaq] = &[
    crate::BuiltinDocumentationFaq { question: "What does `not` return?", answer: "A scalar operand produces a logical scalar; an array produces a logical array; a tabular operand retains its table or timetable container with logical variables." },
    crate::BuiltinDocumentationFaq { question: "How are NaN and complex values interpreted?", answer: "NaN is nonzero. A complex value is true when either component is nonzero. `not` returns false for either case." },
    crate::BuiltinDocumentationFaq { question: "What happens to empty arrays?", answer: "The result is an empty logical array with the same shape." },
    crate::BuiltinDocumentationFaq { question: "Are integers converted to double?", answer: "No. Integer zero/nonzero truth is read directly from each fixed-width class." },
    crate::BuiltinDocumentationFaq { question: "How are tables handled?", answer: "The operation is applied variable by variable and retains the table or timetable class and metadata." },
    crate::BuiltinDocumentationFaq { question: "Can execution remain on a GPU?", answer: "Yes when an exact owning provider supports logical negation. Exact fallback preserves explicit gpuArray intent after validating the restored result." },
];
const LINKS: &[crate::BuiltinDocumentationLink] = &[
    crate::BuiltinDocumentationLink { label: "and", target: crate::BuiltinDocumentationLinkTarget::Builtin("and") },
    crate::BuiltinDocumentationLink { label: "or", target: crate::BuiltinDocumentationLinkTarget::Builtin("or") },
    crate::BuiltinDocumentationLink { label: "xor", target: crate::BuiltinDocumentationLinkTarget::Builtin("xor") },
    crate::BuiltinDocumentationLink { label: "all", target: crate::BuiltinDocumentationLinkTarget::Builtin("all") },
    crate::BuiltinDocumentationLink { label: "any", target: crate::BuiltinDocumentationLinkTarget::Builtin("any") },
    crate::BuiltinDocumentationLink { label: "Implementation", target: crate::BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/bit/not.rs") },
];
const EVIDENCE: crate::BuiltinDocumentationEvidence = crate::BuiltinDocumentationEvidence {
    implementation: &[crate::BuiltinDocumentationLink { label: "Logical negation runtime", target: crate::BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/bit/not.rs") }],
    verification: &[
        crate::BuiltinEvidenceReference { kind: crate::BuiltinEvidenceKind::UnitTest, label: "Scalar, array, integer, character, complex, tabular, and error behavior", location: "crates/runmat-runtime/src/builtins/logical/bit/not/tests.rs" },
        crate::BuiltinEvidenceReference { kind: crate::BuiltinEvidenceKind::WgpuTest, label: "Actual provider execution and residency", location: "crates/runmat-runtime/src/builtins/logical/bit/not/tests.rs" },
    ],
    notes: &[],
};
pub(super) const NOT_DOCUMENTATION: crate::BuiltinDocumentation = crate::BuiltinDocumentation {
    authority: crate::BuiltinDocumentationAuthority::Catalog,
    title: Some("not"),
    slug: Some("not"),
    summary: "Compute element-wise logical negation.",
    description:
        "`not(A)` and `~A` invert the zero/nonzero truth value of every supported element.",
    keywords: &[
        "not",
        "negation",
        "logical",
        "integer",
        "complex",
        "character",
        "table",
        "timetable",
        "gpuArray",
    ],
    related: &["and", "or", "xor", "all", "any", "gpuArray", "gather"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("Before R2006a"),
    status: Some(crate::BuiltinDocumentationStatus::Stable),
};
