//! `geometry.pocket` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "POCKET",
    name: "geometry.pocket",
    positional: &[A::Feature, A::Feature, A::PositiveScalar, A::Direction],
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
            "The profile is swept like `geometry.extrude` to form a cutting tool, which is subtracted from `target`. The profile must realize as a face or sheet so that the sweep is a solid; at realization the tool must also overlap the target volume.",
            "The feature graphs of `target` and `profile` are merged; features they share are kept once. Two graphs that define the same feature identity differently are rejected.",
            DEFERRED_REALIZATION,
            SI_VALUES,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "`geometry.facesIntroducedBy(pocketed)` selects the pocket walls and floor.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:pocket:InvalidInput` reports every failure of the call: the wrong number of arguments, a `target` or `profile` that is not a `geometry.Feature`, conflicting feature definitions, a depth that cannot be converted, or a `direction` that is not a finite three-element real vector, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "record-pocket",
        "Record a 3 mm deep pocket",
        "plate = geometry.box(80e-3, 50e-3, 8e-3);\nprofile = geometry.box(20e-3, 10e-3, 1e-3);\npocketed = geometry.pocket(plate, profile, 3e-3, [0 0 1]);\nclass(pocketed)",
        "assert(isa(pocketed, 'geometry.Feature'));",
    ),
    error_example(
        "zero-depth",
        "A zero depth is rejected",
        "plate = geometry.box(80e-3, 50e-3, 8e-3);\nprofile = geometry.box(20e-3, 10e-3, 1e-3);\npocketed = geometry.pocket(plate, profile, 0, [0 0 1]);",
        "RunMat:geometry:pocket:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Can the profile be a solid?",
        answer: "No. The sweep of the profile must be a solid tool, so the profile must be a face or sheet. Use `geometry.subtract` to remove a solid tool directly.",
    },
    UNITS_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.pocket",
    summary: "Cut a pocket by sweeping a face profile into a geometry feature.",
    description: "`feature = geometry.pocket(target, profile, depth, direction)` returns a `geometry.Feature` for `target` with the linear sweep of `profile` by `depth` meters along `direction` removed.",
    keywords: &["geometry", "cad", "pocket", "cut", "recess", "sweep", "profile", "feature"],
    related: &["geometry.extrude", "geometry.hole", "geometry.subtract", "geometry.facesIntroducedBy"],
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
    "geometry.pocket",
    DOCUMENTATION,
    &GEOMETRY_POCKET_DESCRIPTOR,
    infer,
);
