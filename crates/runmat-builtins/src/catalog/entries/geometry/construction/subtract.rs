//! `geometry.subtract` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "SUBTRACT",
    name: "geometry.subtract",
    positional: &[A::Feature],
    variadic: Some((A::Feature, 1)),
    result: FEATURE_RESULT,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_geometry(request, entry, SIGNATURE)
}

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "One base and at least one tool are required. All tools are removed in one operation. The feature graphs of all operands are merged so that shared features are kept once; conflicting definitions of one feature identity are rejected.",
            "A tool that misses the base leaves the base unchanged when realized. Use `geometry.hole` or `geometry.pocket` when a cut must overlap the part.",
            DEFERRED_REALIZATION,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "`geometry.facesIntroducedBy(body)` on the subtraction selects the faces the tools created, such as bore walls.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:subtract:InvalidInput` reports every failure of the call: the wrong number of arguments, a call with no tool, an operand that is not a `geometry.Feature`, or conflicting feature definitions, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "bored-bracket",
        "Bore a bracket and name its regions",
        "stock = geometry.box(80e-3, 50e-3, 8e-3);\nbore = geometry.cylinder(5e-3, 8e-3, [0 0 1], [40e-3 25e-3 0]);\nbody = geometry.subtract(stock, bore);\npart = geometry.finish(body);\npart = geometry.withRegion(part, \"body\", geometry.solid(body));\npart = geometry.withRegion(part, \"bore\", geometry.facesIntroducedBy(body));\nclass(part)",
        "assert(isa(body, 'geometry.Feature')); assert(isa(part, 'geometry.Generated'));",
    ),
    error_example(
        "no-tool",
        "A base without a tool is rejected",
        "stock = geometry.box(80e-3, 50e-3, 8e-3);\nbody = geometry.subtract(stock);",
        "RunMat:geometry:subtract:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[REALIZATION_FAQ, UNITS_FAQ];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.subtract",
    summary: "Remove one or more tool features from a base geometry feature.",
    description: "`feature = geometry.subtract(base, tool1, tool2, ...)` returns a `geometry.Feature` for `base` with the volume of every tool removed.",
    keywords: &["geometry", "cad", "subtract", "cut", "difference", "boolean", "feature"],
    related: &["geometry.union", "geometry.intersect", "geometry.hole", "geometry.pocket", "geometry.facesIntroducedBy"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[COMBINE_SOURCE, GRAPH_VALIDATION_SOURCE],
    verification: &[COMPOSITION_TEST],
    notes: &[],
    status: BuiltinDocumentationStatus::Stable,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.subtract",
    DOCUMENTATION,
    &GEOMETRY_SUBTRACT_DESCRIPTOR,
    infer,
);
