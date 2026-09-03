const SECTIONS: &[crate::BuiltinDocumentationSection] = &[
    crate::BuiltinDocumentationSection { heading: "Truth and shape semantics", paragraphs: &["`and(A,B)` converts corresponding elements to zero/nonzero truth values after compatible-size expansion, then returns true where both values are true. Zero is false; every nonzero real value, including NaN, is true. The result is logical and has the expanded shape.", "All eight fixed-width integer classes are tested directly in their authoritative storage. Empty shapes propagate without adding elements."] },
    crate::BuiltinDocumentationSection { heading: "Tables and accelerated execution", paragraphs: &["For tables and timetables, `and` operates on corresponding variables. Variable names, row names, and row times may appear in a different order; RunMat aligns them to the first input before evaluation and retains that input's container metadata.", "A supported operation on inputs owned by the same provider remains resident. Unsupported provider routes gather exact values; explicitly requested residency is restored only after validating owner, device, shape, storage, and aliasing.", "Complex and character operands are available in RunMat mode. The compatibility pin accepts the documented real numeric, logical, table, and timetable forms."] },
];
const EXAMPLES: &[crate::BuiltinExample] = &[
    crate::BuiltinExample { id: "scalar", title: "Combine scalar conditions", program: "tf = and(true, false)", display_output: Some("tf = false"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, false));" } },
    crate::BuiltinExample { id: "array", title: "Combine numeric arrays", program: "A = [1 0 2 0];\nB = [3 4 0 0];\ntf = and(A, B)", display_output: Some("tf = [true false false false]"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([1 0 0 0])));" } },
    crate::BuiltinExample { id: "implicit-expansion", title: "Expand a row across a column", program: "tf = and([1; 0], [2 0 3])", display_output: Some("tf = [true false true; false false false]"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([1 0 1; 0 0 0])));" } },
    crate::BuiltinExample { id: "integer", title: "Evaluate wide integers exactly", program: "x = [uint64(0) intmax('uint64')];\ntf = and(x, uint64(1))", display_output: Some("tf = [false true]"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([0 1])));" } },
    crate::BuiltinExample { id: "table", title: "Apply conjunction to table variables", program: "T = table([0; 2], [1; 0], 'VariableNames', {'A', 'B'});\nR = and(T, 1)", display_output: Some("R is a table with logical variables"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Portable, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(istable(R)); assert(isequal(R.A, logical([0; 1]))); assert(isequal(R.B, logical([1; 0])));" } },
    crate::BuiltinExample { id: "complex-extension", title: "Use complex truth values in RunMat mode", program: "tf = and([0+0i 1+0i], 0+2i)", display_output: Some("tf = [false true]"), compatibility: crate::BuiltinExampleCompatibility::RunMat, harness: crate::BuiltinExampleHarness::Portable, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([0 1])));" } },
    crate::BuiltinExample { id: "gpu", title: "Keep a supported result resident", program: "A = gpuArray([0 2 3]);\nB = gpuArray([1 4 0]);\nG = and(A, B);\ntf = gather(G)", display_output: Some("G remains resident and tf = [false true false]"), compatibility: crate::BuiltinExampleCompatibility::Matlab, harness: crate::BuiltinExampleHarness::Wgpu, verification: crate::BuiltinExampleVerification::Assertions { source: "assert(isa(G, 'gpuArray')); assert(isequal(tf, logical([0 1 0])));" } },
];
const FAQS: &[crate::BuiltinDocumentationFaq] = &[
    crate::BuiltinDocumentationFaq { question: "What does `and` return?", answer: "Scalar operands produce a logical scalar; array operands produce logical arrays; tabular operands retain their table or timetable container with logical variables." },
    crate::BuiltinDocumentationFaq { question: "How is NaN interpreted?", answer: "NaN is nonzero and therefore true before the logical operation is applied." },
    crate::BuiltinDocumentationFaq { question: "Does implicit expansion apply?", answer: "Yes. Singleton dimensions expand against compatible dimensions; incompatible array shapes report a size error." },
    crate::BuiltinDocumentationFaq { question: "Are integers converted to double?", answer: "No. Integer zero/nonzero truth is read directly from each fixed-width class." },
    crate::BuiltinDocumentationFaq { question: "How are tables handled?", answer: "The operation is applied variable by variable after aligning compatible tabular metadata. The result retains the first tabular input's class and metadata." },
    crate::BuiltinDocumentationFaq { question: "Can execution remain on a GPU?", answer: "Yes when an exact owning provider supports the operation. Exact fallback preserves explicit gpuArray intent after validating the restored result." },
];
const LINKS: &[crate::BuiltinDocumentationLink] = &[
    crate::BuiltinDocumentationLink { label: "and", target: crate::BuiltinDocumentationLinkTarget::Builtin("and") },
    crate::BuiltinDocumentationLink { label: "or", target: crate::BuiltinDocumentationLinkTarget::Builtin("or") },
    crate::BuiltinDocumentationLink { label: "xor", target: crate::BuiltinDocumentationLinkTarget::Builtin("xor") },
    crate::BuiltinDocumentationLink { label: "not", target: crate::BuiltinDocumentationLinkTarget::Builtin("not") },
    crate::BuiltinDocumentationLink { label: "gpuArray", target: crate::BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    crate::BuiltinDocumentationLink { label: "Implementation", target: crate::BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/bit/and.rs") },
];
const EVIDENCE: crate::BuiltinDocumentationEvidence = crate::BuiltinDocumentationEvidence {
    implementation: &[crate::BuiltinDocumentationLink { label: "conjunction runtime", target: crate::BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/bit/and.rs") }],
    verification: &[
        crate::BuiltinEvidenceReference { kind: crate::BuiltinEvidenceKind::UnitTest, label: "Scalar, array, integer, character, complex, tabular, broadcast, and error behavior", location: "crates/runmat-runtime/src/builtins/logical/bit/and/tests.rs" },
        crate::BuiltinEvidenceReference { kind: crate::BuiltinEvidenceKind::WgpuTest, label: "Actual provider execution and residency", location: "crates/runmat-runtime/src/builtins/logical/bit/and/tests.rs" },
    ],
    notes: &[],
};
pub(super) const AND_DOCUMENTATION: crate::BuiltinDocumentation = crate::BuiltinDocumentation {
    authority: crate::BuiltinDocumentationAuthority::Catalog,
    title: Some("and"),
    slug: Some("and"),
    summary: "Compute element-wise logical conjunction.",
    description:
        "`and(A,B)` and `A & B` return true where both corresponding operands are nonzero.",
    keywords: &[
        "and",
        "conjunction",
        "logical",
        "integer",
        "table",
        "timetable",
        "gpuArray",
    ],
    related: &[
        "and", "or", "xor", "not", "all", "any", "gpuArray", "gather",
    ],
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
