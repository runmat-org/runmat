//! `geometry.revolve` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "REVOLVE",
    name: "geometry.revolve",
    positional: &[A::Feature, A::Direction, A::Point, A::PositiveScalar],
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
            "`axis` is a nonzero three-element direction and `origin` is a point on the axis in meters. `angle` is in radians, must be positive, and may be at most `2*pi` for a full revolution. Construction checks that the angle is positive; the `2*pi` upper bound is checked when the graph is realized.",
            "The profile must realize as a face or sheet; revolving a face produces a solid. A solid profile, such as a box, is accepted at construction but rejected at realization. The programmatic builder does not yet provide sketch or planar-face constructors.",
            DEFERRED_REALIZATION,
            SI_VALUES,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "`geometry.facesIntroducedBy(revolution)` selects the swept faces, for example the outer surface of a turned part.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:revolve:InvalidInput` reports every failure of the call: the wrong number of arguments, a `profile` that is not a `geometry.Feature`, an `axis` or `origin` that is not a finite three-element real vector, or an angle that cannot be converted, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "half-turn",
        "Record a half revolution about the y axis",
        "profile = geometry.box(10e-3, 1e-3, 20e-3);\nturned = geometry.revolve(profile, [0 1 0], [-20e-3 0 0], pi);\nclass(turned)",
        "assert(isa(turned, 'geometry.Feature'));",
    ),
    error_example(
        "negative-angle",
        "A negative angle is rejected",
        "profile = geometry.box(10e-3, 1e-3, 20e-3);\nturned = geometry.revolve(profile, [0 1 0], [0 0 0], -pi);",
        "RunMat:geometry:revolve:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Is the angle in degrees?",
        answer: "No. It is in radians; use `deg2rad(90)` or `pi/2` for a quarter turn.",
    },
    UNITS_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.revolve",
    summary: "Sweep a face profile around an axis to make a solid of revolution.",
    description: "`feature = geometry.revolve(profile, axis, origin, angle)` returns a `geometry.Feature` for the rotation of `profile` by `angle` radians about the line through `origin` along `axis`.",
    keywords: &["geometry", "cad", "revolve", "revolution", "lathe", "sweep", "profile", "feature"],
    related: &["geometry.extrude", "geometry.pocket", "geometry.union", "geometry.facesIntroducedBy"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[FEATURES_SOURCE, GRAPH_VALIDATION_SOURCE],
    verification: &[FULL_SURFACE_TEST, OCCT_PROFILE_TEST],
    notes: &["Profiles reach exact sweeps through face or sheet geometry; the programmatic builder has no planar-profile constructor yet."],
    status: BuiltinDocumentationStatus::Experimental,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.revolve",
    DOCUMENTATION,
    &GEOMETRY_REVOLVE_DESCRIPTOR,
    infer,
);
