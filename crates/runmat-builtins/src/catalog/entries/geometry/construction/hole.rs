//! `geometry.hole` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "HOLE",
    name: "geometry.hole",
    positional: &[
        A::Feature,
        A::PositiveScalar,
        A::PositiveScalar,
        A::Direction,
        A::Point,
    ],
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
            "The cutting tool is a cylinder whose base circle is centered at `origin` and which extends `depth` meters along `axis`. Put `origin` on the entry face and point `axis` into the material; a depth equal to the wall thickness gives a through hole.",
            "When the graph is realized, the tool must overlap the target volume; a hole that misses the part fails with the target and tool bounds rather than producing an unchanged part.",
            DEFERRED_REALIZATION,
            SI_VALUES,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "`geometry.facesIntroducedBy(holed)` selects the bore wall and any hole floor, a common region for bolt loads, pins, or fixed supports.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:hole:InvalidInput` reports every failure of the call: the wrong number of arguments, a `target` that is not a `geometry.Feature`, a dimension that cannot be converted, or an `axis` or `origin` that is not a finite three-element real vector, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "through-hole",
        "Drill an 8 mm through hole and select its wall",
        "plate = geometry.box(80e-3, 50e-3, 8e-3);\nholed = geometry.hole(plate, 4e-3, 8e-3, [0 0 1], [40e-3 25e-3 0]);\nboreWall = geometry.facesIntroducedBy(holed);\nclass(holed)",
        "assert(isa(holed, 'geometry.Feature')); assert(isa(boreWall, 'geometry.RegionOutput'));",
    ),
    error_example(
        "zero-radius",
        "A zero radius is rejected",
        "plate = geometry.box(80e-3, 50e-3, 8e-3);\nholed = geometry.hole(plate, 0, 8e-3, [0 0 1], [40e-3 25e-3 0]);",
        "RunMat:geometry:hole:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How is `geometry.hole` different from subtracting a cylinder?",
        answer: "The cut is the same. `geometry.hole` records it as one named operation on the target and checks at realization that the tool overlaps the part.",
    },
    UNITS_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.hole",
    summary: "Cut a cylindrical hole into a geometry feature.",
    description: "`feature = geometry.hole(target, radius, depth, axis, origin)` returns a `geometry.Feature` for `target` with a cylinder of `radius` and `depth` removed, starting at `origin` and running along `axis`.",
    keywords: &["geometry", "cad", "hole", "bore", "drill", "cut", "feature"],
    related: &["geometry.cylinder", "geometry.subtract", "geometry.pocket", "geometry.facesIntroducedBy", "geometry.instance"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[FEATURES_SOURCE, GRAPH_VALIDATION_SOURCE],
    verification: &[FULL_SURFACE_TEST],
    notes: &[],
    status: BuiltinDocumentationStatus::Stable,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.hole",
    DOCUMENTATION,
    &GEOMETRY_HOLE_DESCRIPTOR,
    infer,
);
