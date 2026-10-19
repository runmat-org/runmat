use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime parameter-tree traversal",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/training/dlupdate/mod.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dlupdate"),
    slug: Some("dlupdate"),
    summary: "Apply a function handle to matching parameter-tree leaves.",
    description: "`dlupdate(fun, params, A1, ...)` invokes `fun` for each leaf of matching parameter trees and reconstructs one output tree for each callback result.",
    keywords: &["dlupdate", "deep learning", "model update", "parameter tree", "function handle"],
    related: &["dlarray", "dlfeval", "dlgradient", "dlnetwork", "adamupdate"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Traversal and outputs",
            paragraphs: &[
                "The first argument must be a callable function handle or closure. Numeric values and dlarray objects are leaves. Cells, structures, and structure arrays are traversed recursively, and all supplied trees must have matching container geometry and ordered fields. The requested output count is forwarded to every leaf callback; each output is rebuilt with the original container shape.",
                "Independent Learnables structures, tables, and timetables preserve their Layer and Parameter metadata and apply the callback only to the Value column. A dlnetwork projects its parameter table from canonical Layers, validates all input networks before callbacks, and applies accepted callback values back to Layers. Arbitrary custom objects do not have a traversal contract and return Unsupported.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Effects and placement",
            paragraphs: &[
                "The function does not gather a tree before invoking its callback. A callback may read or change ambient state, suspend, access providers, or return resident values. The result's numerical class and residency therefore depend on the callback; a static type fact about the input leaf is not proof of the output leaf's type.",
                "Traversal is bounded by the runtime's supported nesting depth. Callback errors and mismatched output counts stop the traversal. This in-memory implementation does not require filesystem or ONNX loading.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "struct-update",
        title: "Update matching struct leaves",
        program: "params.w = 2; params.b = 3; delta.w = 1; delta.b = 4; updated = dlupdate(@plus, params, delta); fprintf('%d %d\\n', updated.w, updated.b);",
        display_output: Some("3 7"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(updated.w == 3); assert(updated.b == 7);",
        },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does dlupdate gather gpuArray leaves?",
            answer: "No. It forwards leaf values to the callback. A callback can preserve residency, gather, or return a host value; dlupdate itself does not choose that placement.",
        },
        BuiltinDocumentationFaq {
            question: "Which parts of a learnables table are updated?",
            answer: "Only entries in Value are passed to the callback. Layer and Parameter metadata are retained when matching input trees are reconstructed.",
        },
    ],
    links: &[
        SOURCE,
        BuiltinDocumentationLink {
            label: "MATLAB dlupdate reference",
            target: BuiltinDocumentationLinkTarget::External(
                "https://www.mathworks.com/help/deeplearning/ref/dlupdate.html",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Parameter-tree reconstruction and callback output forwarding",
            location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
