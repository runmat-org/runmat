//! `geometry.cylinder` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "CYLINDER",
    name: "geometry.cylinder",
    positional: &[A::PositiveScalar, A::PositiveScalar, A::Direction, A::Point],
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
            "`radius` and `height` are positive lengths in meters. `axis` is a three-element direction; only its direction is used, so `[0 0 1]` and `[0 0 5]` give the same cylinder, but an all-zero axis is rejected. `origin` is the center of the base circle in meters; the opposite face is at `origin + height * axis / norm(axis)`.",
            "Use the cylinder as a boss with `geometry.union`, or as a cutting tool with `geometry.subtract`. For a drilled bore, `geometry.hole` records the same cylindrical cut as one named operation and checks at realization that it overlaps the target.",
            DEFERRED_REALIZATION,
            SI_VALUES,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "When the cylinder is a cutting tool, `geometry.facesIntroducedBy` on the subtraction selects the new bore wall, a typical bolt-load or fixed-support region.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:cylinder:InvalidInput` reports every failure of the call: the wrong number of arguments, a scalar that cannot be converted, or an `axis` or `origin` that is not a finite three-element real vector, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "bore-tool",
        "Cut a 10 mm bore through a plate",
        "plate = geometry.box(80e-3, 50e-3, 8e-3);\n% 5 mm radius, through the 8 mm thickness, centered on the plate\nbore = geometry.cylinder(5e-3, 8e-3, [0 0 1], [40e-3 25e-3 0]);\nbody = geometry.subtract(plate, bore);\nclass(body)",
        "assert(isa(bore, 'geometry.Feature')); assert(isa(body, 'geometry.Feature'));",
    ),
    error_example(
        "zero-axis",
        "An all-zero axis is rejected",
        "pin = geometry.cylinder(3e-3, 12e-3, [0 0 0], [0 0 0]);",
        "RunMat:geometry:cylinder:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Is `origin` the center of the cylinder?",
        answer: "No. It is the center of the base circle. The cylinder extends `height` from there along `axis`.",
    },
    UNITS_FAQ,
    REALIZATION_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.cylinder",
    summary: "Create an exact right circular cylinder feature from SI dimensions, an axis, and a base center.",
    description: "`feature = geometry.cylinder(radius, height, axis, origin)` returns a `geometry.Feature` for a solid cylinder whose base circle is centered at `origin` and which extends `height` meters along `axis`.",
    keywords: &["geometry", "cad", "cylinder", "boss", "pin", "primitive", "feature"],
    related: &["geometry.box", "geometry.sphere", "geometry.hole", "geometry.subtract", "geometry.union", "geometry.instance"],
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
    "geometry.cylinder",
    DOCUMENTATION,
    &GEOMETRY_CYLINDER_DESCRIPTOR,
    infer,
);
