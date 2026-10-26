//! `geometry.union` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "UNION",
    name: "geometry.union",
    positional: &[],
    variadic: Some((A::Feature, 2)),
    result: FEATURE_RESULT,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_geometry(request, entry, SIGNATURE)
}

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "At least two operands are required. Their feature graphs are merged so that shared features are kept once; two graphs that define the same feature identity differently are rejected. The union is one operation on all operands, not a chain of pairwise unions.",
            "Operands that touch or overlap fuse into one solid. Disjoint operands remain separate solids in one result, so `geometry.solid` (which requires exactly one solid) fails at realization for a disjoint union.",
            DEFERRED_REALIZATION,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "Fuse bosses, ribs, and stock into one body before selecting its material region with `geometry.solid`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:union:InvalidInput` reports every failure of the call: the wrong number of arguments, fewer than two operands, an operand that is not a `geometry.Feature`, or conflicting feature definitions, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "plate-with-boss",
        "Fuse a boss onto a plate",
        "plate = geometry.box(80e-3, 50e-3, 8e-3);\nboss = geometry.cylinder(8e-3, 12e-3, [0 0 1], [40e-3 25e-3 8e-3]);\nbody = geometry.union(plate, boss);\nclass(body)",
        "assert(isa(body, 'geometry.Feature'));",
    ),
    error_example(
        "one-operand",
        "A single operand is rejected",
        "plate = geometry.box(80e-3, 50e-3, 8e-3);\nbody = geometry.union(plate);",
        "RunMat:geometry:union:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does the order of operands matter?",
        answer: "Not for the shape. The operand order is recorded in the operation, so it does affect the feature identity.",
    },
    REALIZATION_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.union",
    summary: "Fuse two or more geometry features into one exact Boolean union.",
    description: "`feature = geometry.union(feature1, feature2, ...)` returns a `geometry.Feature` for the volume occupied by any operand.",
    keywords: &["geometry", "cad", "union", "fuse", "boolean", "combine", "feature"],
    related: &["geometry.subtract", "geometry.intersect", "geometry.instance", "geometry.solid"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[COMBINE_SOURCE, GRAPH_VALIDATION_SOURCE],
    verification: &[BOOLEAN_TEST],
    notes: &[],
    status: BuiltinDocumentationStatus::Stable,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.union",
    DOCUMENTATION,
    &GEOMETRY_UNION_DESCRIPTOR,
    infer,
);
