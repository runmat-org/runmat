use crate::*;

const COMBVEC_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime combvec implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/sequences/combvec.rs",
    ),
};
const PAD_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime padsequences implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/sequences/pad/mod.rs",
    ),
};

pub(super) const COMBVEC: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("combvec"),
    slug: Some("combvec"),
    summary: "Generate all column combinations from numeric matrices.",
    description: "`combvec(A,B,...)` returns a host double matrix whose columns are all combinations of the input matrix columns. The first input varies fastest.",
    keywords: &["combvec", "deep learning", "neural network", "cartesian product"],
    related: &["padsequences", "dlarray"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and output", paragraphs: &[
            "Supply at least one finite real numeric scalar or two-dimensional matrix. The output has the sum of input row counts and the product of input column counts. Each input needs at least one row and column. In RunMat compatibility mode, typed integer inputs are also accepted; the result is double, so wide integers may round at this floating-point output boundary. MATLAB compatibility mode does not admit this unverified integer form.",
            "The implementation rejects more than 1,000,000 output columns or 10,000,000 output elements before allocation.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "`combvec` constructs a host matrix. Resident numeric inputs are gathered after host-verifiable argument admission. The Cartesian product has no GPU kernel or fused execution path in this implementation.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "vectors", title: "Combine vectors", program: "M = combvec([1 2], [10 20 30]); fprintf('%d %d\\n', size(M,1), size(M,2));", display_output: Some("2 6"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(M), [2 6])); assert(isequal(M(:,1), [1;10])); assert(isequal(M(:,2), [2;10]));" } },
        BuiltinExample { id: "matrix-columns", title: "Stack matrix columns", program: "M = combvec([1 3; 2 4], [10 20]); fprintf('%d %d\\n', size(M,1), size(M,2));", display_output: Some("3 4"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(M), [3 4])); assert(isequal(M(:,1), [1;2;10]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which input varies fastest?", answer: "The first input varies fastest. Its first column is paired with the first column of every later input before that later input advances." },
        BuiltinDocumentationFaq { question: "Does combvec keep integer storage?", answer: "No. Its result is a double matrix; input integers are converted at the Cartesian output boundary." },
    ],
    links: &[COMBVEC_SOURCE, BuiltinDocumentationLink { label: "padsequences", target: BuiltinDocumentationLinkTarget::Builtin("padsequences") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[COMBVEC_SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Column combination order and integer input", location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs" }],
        notes: &["Host output and resident gather behavior need integrated native/browser qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const PADSEQUENCES: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("padsequences"),
    slug: Some("padsequences"),
    summary: "Pad or truncate numeric sequence cell arrays.",
    description: "`padsequences(sequences,paddingDim)` pads or truncates numeric arrays in a cell array along a one-based dimension. It returns a host tensor by default or a cell array when `UniformOutput` is false; request `[Y,mask]` for original-data positions.",
    keywords: &["padsequences", "deep learning", "sequence", "padding"],
    related: &["combvec", "sequenceInputLayer", "dlarray"],
    sections: &[
        BuiltinDocumentationSection { heading: "Padding options", paragraphs: &[
            "`PaddingDirection` (or `Direction`) accepts `left`, `right`, or `both`; the default is `right`. `Length` accepts `longest`, `shortest`, or a positive integer. `PaddingValue` accepts a finite numeric scalar and defaults to zero. `UniformOutput` defaults to true and stacks equal non-padding dimensions along one final sequence axis; false preserves the input cell shape. Homogeneous sequence classes are preserved, including exact wide integer storage. Mixed numeric classes produce double when stacked; each original class is retained in cell output.",
            "Sequences must be finite real numeric scalars or arrays. The requested dimension is a positive integer. For uniform output, all non-padding dimensions must match. The implementation rejects more than 10,000,000 elements in an output sequence or stacked tensor.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Padding currently materializes host typed tensors. It does not run a resident GPU padding kernel. Resident numeric elements inside the input cell are selectively gathered from their owning provider after host options, host elements, and known shape compatibility have been checked. Homogeneous input classes survive that transfer. A second output is a logical mask with the same tensor or cell-container shape as the padded data.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "right-padding", title: "Right-pad two sequences", program: "Y = padsequences({[1 2], 3}, 2, 'PaddingValue', -1); fprintf('%d %d %d\\n', size(Y,1), size(Y,2), size(Y,3));", display_output: Some("1 2 2"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Y), [1 2 2])); assert(isequal(Y(:).', [1 2 3 -1]));" } },
        BuiltinExample { id: "cell-output", title: "Preserve cell shape", program: "C = padsequences({[1 2 3], [4 5]}, 2, 'Length', 2, 'PaddingDirection', 'left', 'UniformOutput', false); first = C{1}; fprintf('%d %d\\n', first(1), first(2));", display_output: Some("2 3"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(C)); assert(isequal(size(C), [1 2])); assert(isequal(C{1}, [2 3]));" } },
        BuiltinExample { id: "mask", title: "Identify original samples", program: "[Y, mask] = padsequences({[1 2], 3}, 2); fprintf('%d\\n', sum(mask(:)));", display_output: Some("3"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(mask)); assert(isequal(size(mask), size(Y))); assert(sum(mask(:)) == 3); assert(mask(1,2,2) == false);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How is the target length chosen?", answer: "It is the longest sequence length by default. Select `Length` as `shortest` or a positive integer to change it." },
        BuiltinDocumentationFaq { question: "When should I use UniformOutput=false?", answer: "Use it when the non-padding dimensions differ or when you need the result in cells with the same shape as the input cell array." },
        BuiltinDocumentationFaq { question: "Does padsequences preserve integer arrays?", answer: "Yes for a cell collection with one numeric class. Mixed-class collections produce double when stacked; with UniformOutput=false each cell retains its original class. The optional mask is logical." },
        BuiltinDocumentationFaq { question: "Are symmetric padding and categorical arrays supported?", answer: "No. This implementation accepts finite real numeric arrays and constant numeric padding; symmetric reflection and categorical sequence forms are not implemented." },
    ],
    links: &[PAD_SOURCE, BuiltinDocumentationLink { label: "sequenceInputLayer", target: BuiltinDocumentationLinkTarget::Builtin("sequenceInputLayer") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[PAD_SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Uniform and cell padding behavior", location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs" }],
        notes: &["Native/browser and provider behavior require integrated qualification; GPU sequence padding is not implemented."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
