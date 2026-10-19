//! `geometry.box` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "BOX",
    name: "geometry.box",
    positional: &[A::PositiveScalar, A::PositiveScalar, A::PositiveScalar],
    variadic: None,
    result: FEATURE_RESULT,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_geometry(request, entry, SIGNATURE)
}

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The box occupies `[0, width] x [0, depth] x [0, height]`. Place it elsewhere with `geometry.transform`, combine it with `geometry.union`, `geometry.subtract`, or `geometry.intersect`, or cut it with `geometry.hole` and `geometry.pocket`.",
            DEFERRED_REALIZATION,
            SI_VALUES,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "A box is the usual stock for plates, brackets, and blocks. Cut features from it, then publish `geometry.solid(body)` as the material region and `geometry.facesIntroducedBy` selections as boundary regions.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:box:InvalidInput` reports every failure of the call: the wrong number of arguments, a dimension that is not a real numeric scalar, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "plate-stock",
        "Create mounting-plate stock",
        "% 80 mm x 50 mm x 8 mm plate, entered in meters\nplate = geometry.box(80e-3, 50e-3, 8e-3);\nclass(plate)",
        "assert(isa(plate, 'geometry.Feature'));",
    ),
    error_example(
        "zero-thickness",
        "A zero dimension is rejected",
        "plate = geometry.box(80e-3, 50e-3, 0);",
        "RunMat:geometry:box:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    UNITS_FAQ,
    BuiltinDocumentationFaq {
        question: "Where is the box placed?",
        answer: "One corner is at the origin and the box extends along +x, +y, and +z. Use `geometry.transform` with a 4-by-4 homogeneous matrix to move or rotate it.",
    },
    REALIZATION_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.box",
    summary: "Create an exact rectangular box feature from SI dimensions.",
    description: "`feature = geometry.box(width, depth, height)` returns a `geometry.Feature` for an axis-aligned rectangular solid with one corner at the origin and extents `width` along +x, `depth` along +y, and `height` along +z, in meters.",
    keywords: &["geometry", "cad", "box", "block", "primitive", "feature"],
    related: &["geometry.cylinder", "geometry.sphere", "geometry.transform", "geometry.subtract", "geometry.hole", "geometry.finish"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[PRIMITIVES_SOURCE, GRAPH_VALIDATION_SOURCE],
    verification: &[COMPOSITION_TEST, ERROR_OWNER_TEST, INTEGER_TEST],
    notes: &[],
    status: BuiltinDocumentationStatus::Stable,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.box",
    DOCUMENTATION,
    &GEOMETRY_BOX_DESCRIPTOR,
    infer,
);
