//! `geometry.intersect` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "INTERSECT",
    name: "geometry.intersect",
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
            "At least two operands are required. Their feature graphs are merged so that shared features are kept once; conflicting definitions of one feature identity are rejected. The first operand is the argument of the Boolean and the rest are its tools.",
            "If the operands do not overlap, the realized result is empty and region selections on it fail their cardinality checks.",
            DEFERRED_REALIZATION,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "Intersection is a compact way to trim a primitive, for example a sphere clipped by a cylinder to make a dome.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:intersect:InvalidInput` reports every failure of the call: the wrong number of arguments, fewer than two operands, an operand that is not a `geometry.Feature`, or conflicting feature definitions, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "lens",
        "Intersect two spheres",
        "a = geometry.sphere(20e-3, [0 0 0]);\nb = geometry.sphere(20e-3, [25e-3 0 0]);\nlens = geometry.intersect(a, b);\nclass(lens)",
        "assert(isa(lens, 'geometry.Feature'));",
    ),
    error_example(
        "one-operand",
        "A single operand is rejected",
        "a = geometry.sphere(20e-3, [0 0 0]);\nlens = geometry.intersect(a);",
        "RunMat:geometry:intersect:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[REALIZATION_FAQ];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.intersect",
    summary: "Keep only the volume common to two or more geometry features.",
    description: "`feature = geometry.intersect(feature1, feature2, ...)` returns a `geometry.Feature` for the volume that lies inside every operand.",
    keywords: &["geometry", "cad", "intersect", "common", "boolean", "feature"],
    related: &["geometry.union", "geometry.subtract", "geometry.sphere", "geometry.box"],
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
    "geometry.intersect",
    DOCUMENTATION,
    &GEOMETRY_INTERSECT_DESCRIPTOR,
    infer,
);
