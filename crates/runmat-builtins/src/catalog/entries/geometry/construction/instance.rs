//! `geometry.instance` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "INSTANCE",
    name: "geometry.instance",
    positional: &[A::Feature],
    variadic: Some((A::Transform, 1)),
    result: FEATURE_RESULT,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_geometry(request, entry, SIGNATURE)
}

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Each transform follows the `geometry.transform` convention: `T(1:3,1:3)` is the linear part, `T(1:3,4)` is the translation in meters, and the final row must be `[0 0 0 1]`; every transform is checked when the call is made. At least one transform is required, and the transforms are kept in call order.",
            "The result holds only the transformed copies. The untransformed `source` is not included unless one of the transforms is `eye(4)`. Combine the copies with other bodies using `geometry.union` or `geometry.subtract`, for example to cut a hole pattern.",
            DEFERRED_REALIZATION,
            SI_VALUES,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "Patterned features keep one feature identity, so `geometry.facesIntroducedBy` on a subtraction that uses the pattern selects every patterned bore wall at once.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:instance:InvalidInput` reports every failure of the call: the wrong number of arguments, a `source` that is not a `geometry.Feature`, a call with no transform, or a transform that is not a finite real 4-by-4 matrix with final row `[0 0 0 1]`, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "pin-pattern",
        "Cut a two-hole pattern",
        "plate = geometry.box(80e-3, 50e-3, 8e-3);\npin = geometry.cylinder(3e-3, 8e-3, [0 0 1], [0 0 0]);\nT1 = eye(4); T1(1:3, 4) = [10e-3; 25e-3; 0];\nT2 = eye(4); T2(1:3, 4) = [70e-3; 25e-3; 0];\npins = geometry.instance(pin, T1, T2);\nbody = geometry.subtract(plate, pins);\nclass(pins)",
        "assert(isa(pins, 'geometry.Feature')); assert(isa(body, 'geometry.Feature'));",
    ),
    error_example(
        "no-transform",
        "At least one transform is required",
        "pin = geometry.cylinder(3e-3, 8e-3, [0 0 1], [0 0 0]);\npins = geometry.instance(pin);",
        "RunMat:geometry:instance:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Is the original feature part of the result?",
        answer: "No. Only the transformed copies are. Include `eye(4)` as one of the transforms to keep a copy in place.",
    },
    UNITS_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.instance",
    summary: "Place one or more transformed copies of a geometry feature.",
    description: "`feature = geometry.instance(source, T1, T2, ...)` returns a `geometry.Feature` whose result contains one copy of `source` per 4-by-4 homogeneous affine transform.",
    keywords: &["geometry", "cad", "instance", "pattern", "copy", "occurrence", "array"],
    related: &["geometry.transform", "geometry.union", "geometry.subtract", "geometry.cylinder"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[PRIMITIVES_SOURCE, GRAPH_VALIDATION_SOURCE],
    verification: &[INSTANCE_TEST, AFFINE_TEST],
    notes: &[],
    status: BuiltinDocumentationStatus::Stable,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.instance",
    DOCUMENTATION,
    &GEOMETRY_INSTANCE_DESCRIPTOR,
    infer,
);
