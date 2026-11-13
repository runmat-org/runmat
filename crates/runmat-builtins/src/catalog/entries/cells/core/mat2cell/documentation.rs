use crate::*;

use super::examples::EXAMPLES;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Partition vectors", paragraphs: &["`mat2cell(A, dim1dist, dim2dist, ...)` divides `A` into contiguous blocks. Each partition vector contains non-negative integer block sizes whose sum equals the corresponding extent of `A`.", "Omitted trailing dimensions remain intact. Zero sizes produce empty blocks while preserving the cell grid."] },
    BuiltinDocumentationSection { heading: "Types and storage", paragraphs: &["Numeric, complex, logical, string, and character arrays are supported. Each block preserves the source class, including exact fixed-width integer storage.", "In RunMat compatibility mode, partition vectors may use any fixed-width integer class. MATLAB compatibility mode accepts the documented floating numeric form."] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &["The result is a host cell array whose shape records the number of blocks per dimension. Automatically resident inputs gather before partitioning; explicit `gpuArray` inputs and partition vectors are rejected until providers expose a block-splitting contract.", "Each result cell owns its value-semantic block, so later mutation does not change `A` or another block."] },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Must partition sizes sum exactly?", answer: "Yes. Each supplied vector must sum to the corresponding input extent." },
    BuiltinDocumentationFaq { question: "What happens when trailing partition vectors are omitted?", answer: "Each omitted dimension remains one complete block, so its input extent is preserved within every output cell." },
    BuiltinDocumentationFaq { question: "Are zero-sized blocks supported?", answer: "Yes. A zero partition entry creates an empty block of the source class while retaining its position in the output cell grid." },
    BuiltinDocumentationFaq { question: "Which input arrays are supported?", answer: "Numeric, complex, logical, string, and character arrays are supported. Cell, structure, and object arrays are not currently supported." },
    BuiltinDocumentationFaq { question: "Does mat2cell copy its input data?", answer: "Yes. Each result cell owns a value-semantic block, so changing one block does not change the source array or another block." },
    BuiltinDocumentationFaq { question: "Can mat2cell partition N-dimensional arrays?", answer: "Yes. Supply one partition vector for each dimension that should be split; any remaining trailing dimensions stay intact." },
    BuiltinDocumentationFaq { question: "How does mat2cell differ from num2cell?", answer: "mat2cell uses explicit contiguous block sizes. num2cell creates scalar cells unless dimensions are grouped." },
    BuiltinDocumentationFaq { question: "How can the blocks be joined again?", answer: "Use cell2mat when the cell grid contains compatible array blocks whose extents form a valid rectangular layout." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "cell2mat",
        target: BuiltinDocumentationLinkTarget::Builtin("cell2mat"),
    },
    BuiltinDocumentationLink {
        label: "num2cell",
        target: BuiltinDocumentationLinkTarget::Builtin("num2cell"),
    },
    BuiltinDocumentationLink {
        label: "cell",
        target: BuiltinDocumentationLinkTarget::Builtin("cell"),
    },
    BuiltinDocumentationLink {
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
    BuiltinDocumentationLink {
        label: "cellfun",
        target: BuiltinDocumentationLinkTarget::Builtin("cellfun"),
    },
    BuiltinDocumentationLink {
        label: "cellstr",
        target: BuiltinDocumentationLinkTarget::Builtin("cellstr"),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/cells/core/mat2cell") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Partition, class, shape, error, and provider behavior", location: "builtins::cells::core::mat2cell::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference and diagnostics", location: "catalog::entries::cells::core::mat2cell::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &["Compatible behavior was checked against the public mat2cell reference for MATLAB R2026a."],
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mat2cell"),
    slug: Some("mat2cell"),
    summary: "Split an array into contiguous cell-array blocks.",
    description:
        "`mat2cell` partitions an array according to one size vector per selected dimension.",
    keywords: &["mat2cell", "cell array", "partition", "block slicing"],
    related: &["cell2mat", "num2cell", "cell"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("before R2006a"),
    status: Some(BuiltinDocumentationStatus::Stable),
};
