//! `geometry.sphere` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "SPHERE",
    name: "geometry.sphere",
    positional: &[A::PositiveScalar, A::Point],
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
            "`radius` is a positive length in meters and `center` is a finite three-element point in meters. Combine spheres with the Boolean builders, for example `geometry.intersect` with a cylinder to make a dome, or `geometry.subtract` to make a spherical seat.",
            DEFERRED_REALIZATION,
            SI_VALUES,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:sphere:InvalidInput` reports every failure of the call: the wrong number of arguments, a radius that cannot be converted, or a `center` that is not a finite three-element real vector, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "dome",
        "Make a 20 mm hemispherical dome",
        "ball = geometry.sphere(20e-3, [0 0 0]);\n% keep the half above z = 0\nslab = geometry.cylinder(20e-3, 20e-3, [0 0 1], [0 0 0]);\ndome = geometry.intersect(ball, slab);\nclass(dome)",
        "assert(isa(ball, 'geometry.Feature')); assert(isa(dome, 'geometry.Feature'));",
    ),
    error_example(
        "nan-center",
        "A non-finite center is rejected",
        "ball = geometry.sphere(20e-3, [0 0 NaN]);",
        "RunMat:geometry:sphere:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[UNITS_FAQ, REALIZATION_FAQ];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.sphere",
    summary: "Create an exact sphere feature from an SI radius and center.",
    description: "`feature = geometry.sphere(radius, center)` returns a `geometry.Feature` for a solid sphere of `radius` meters centered at the three-element point `center`.",
    keywords: &["geometry", "cad", "sphere", "ball", "primitive", "feature"],
    related: &["geometry.box", "geometry.cylinder", "geometry.intersect", "geometry.subtract", "geometry.union"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[PRIMITIVES_SOURCE, GRAPH_VALIDATION_SOURCE],
    verification: &[BOOLEAN_TEST, ERROR_OWNER_TEST],
    notes: &[],
    status: BuiltinDocumentationStatus::Stable,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.sphere",
    DOCUMENTATION,
    &GEOMETRY_SPHERE_DESCRIPTOR,
    infer,
);
