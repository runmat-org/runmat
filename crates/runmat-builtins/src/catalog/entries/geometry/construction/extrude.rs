//! `geometry.extrude` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "EXTRUDE",
    name: "geometry.extrude",
    positional: &[A::Feature, A::PositiveScalar, A::Direction],
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
            "`distance` is a positive length in meters and `direction` is a nonzero three-element direction; only its direction is used. The sweep is one-sided, starting at the profile.",
            "The profile must realize as a face or sheet; sweeping a face produces a solid. A solid cannot be swept, so a profile built from `geometry.box`, `geometry.cylinder`, or `geometry.sphere` is accepted at construction but rejected when the graph is realized. The programmatic builder does not yet provide sketch or planar-face constructors.",
            DEFERRED_REALIZATION,
            SI_VALUES,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "`geometry.facesIntroducedBy(extrusion)` selects the side and cap faces created by the sweep.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:extrude:InvalidInput` reports every failure of the call: the wrong number of arguments, a `profile` that is not a `geometry.Feature`, a distance that cannot be converted, or a `direction` that is not a finite three-element real vector, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "record-extrusion",
        "Record a 50 mm extrusion along z",
        "profile = geometry.box(20e-3, 10e-3, 1e-3);\nbar = geometry.extrude(profile, 50e-3, [0 0 1]);\nclass(bar)",
        "assert(isa(bar, 'geometry.Feature'));",
    ),
    error_example(
        "zero-direction",
        "An all-zero direction is rejected",
        "profile = geometry.box(20e-3, 10e-3, 1e-3);\nbar = geometry.extrude(profile, 50e-3, [0 0 0]);",
        "RunMat:geometry:extrude:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why does my extrusion of a box fail when the study runs?",
        answer: "A box is a solid, and only faces or sheets can be swept. Construction records the operation; realization rejects the solid profile.",
    },
    UNITS_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.extrude",
    summary: "Sweep a face profile along a straight direction to make a prism.",
    description: "`feature = geometry.extrude(profile, distance, direction)` returns a `geometry.Feature` for the linear sweep of `profile` by `distance` meters along `direction`.",
    keywords: &["geometry", "cad", "extrude", "prism", "sweep", "profile", "feature"],
    related: &["geometry.revolve", "geometry.pocket", "geometry.union", "geometry.facesIntroducedBy"],
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
    "geometry.extrude",
    DOCUMENTATION,
    &GEOMETRY_EXTRUDE_DESCRIPTOR,
    infer,
);
