//! `geometry.transform` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "TRANSFORM",
    name: "geometry.transform",
    positional: &[A::Feature, A::Transform],
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
            "`T` is a 4-by-4 real matrix. `T(1:3,1:3)` is the linear part (rotation, and optionally scaling or shear) and `T(1:3,4)` is the translation in meters; the final row must be `[0 0 0 1]`. A pure translation is `T = eye(4); T(1:3,4) = [dx; dy; dz];`.",
            "At construction the builder requires exactly 16 finite real values; whether the final row is `[0 0 0 1]` is checked when the graph is realized. The result is one new feature that depends on `source`; `source` itself is unchanged and remains usable.",
            DEFERRED_REALIZATION,
            SI_VALUES,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "Use `geometry.transform` to position primitives before Boolean operations, since primitives are created at fixed reference positions. To place several copies of one feature, use `geometry.instance`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:transform:InvalidInput` reports a `source` that is not a `geometry.Feature`, or a matrix that does not contain exactly 16 finite real values.",
            "`RunMat:geometry:feature:InvalidInput` (`RM.GEOMETRY.FEATURE.INVALID_CONTRACT`) reports a value that converts but violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. Either error leaves earlier values unchanged; builder values are immutable.",
            "A scalar argument that is not a real numeric scalar (for example text, a logical, or a vector) is rejected while arguments are converted, before the builder runs.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "translate",
        "Move a plate 100 mm along x",
        "plate = geometry.box(80e-3, 50e-3, 8e-3);\nT = eye(4);\nT(1:3, 4) = [100e-3; 0; 0];\nmoved = geometry.transform(plate, T);\nclass(moved)",
        "assert(isa(moved, 'geometry.Feature'));",
    ),
    error_example(
        "wrong-size",
        "A 3-by-3 matrix is rejected",
        "plate = geometry.box(80e-3, 50e-3, 8e-3);\nmoved = geometry.transform(plate, eye(3));",
        "RunMat:geometry:transform:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How do I rotate a part about the z axis?",
        answer: "Put a rotation in the linear part: `c = cos(theta); s = sin(theta); T = eye(4); T(1:2,1:2) = [c -s; s c];`. Rotation is about the origin, so translate first if you need a different pivot.",
    },
    UNITS_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.transform",
    summary: "Move, rotate, or otherwise affinely transform a geometry feature.",
    description: "`feature = geometry.transform(source, T)` returns a `geometry.Feature` whose result is `source` mapped by the 4-by-4 homogeneous affine transform `T`, where a point `p` moves to `T(1:3,1:3)*p + T(1:3,4)`.",
    keywords: &["geometry", "cad", "transform", "translate", "rotate", "affine", "placement"],
    related: &["geometry.instance", "geometry.box", "geometry.union", "geometry.subtract"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[PRIMITIVES_SOURCE, GRAPH_VALIDATION_SOURCE],
    verification: &[BUILDER_TESTS],
    notes: &[],
    status: BuiltinDocumentationStatus::Experimental,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.transform",
    DOCUMENTATION,
    &GEOMETRY_TRANSFORM_DESCRIPTOR,
    infer,
);
