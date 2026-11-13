//! `geometry.chamfer` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "CHAMFER",
    name: "geometry.chamfer",
    positional: &[A::Feature, A::RegionOutput, A::PositiveScalar],
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
            "The chamfer is symmetric: the same `distance` is measured on both faces that meet at each edge. `edges` must be an edge selection from `geometry.edges` or `geometry.edgesIntroducedBy` and must name a feature in `target`'s graph. At realization every selected edge must exist in the target's topology.",
            "Selections are defined by feature lineage, not by entity index or position. Because the region names what the feature produced, it keeps selecting the intended topology when dimensions change and the part is regenerated, and realization fails rather than silently selecting different topology.",
            DEFERRED_REALIZATION,
            SI_VALUES,
            VALIDATION_BEFORE_RESULT,
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "`geometry.facesIntroducedBy(beveled)` selects the bevel faces.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:chamfer:InvalidInput` reports every failure of the call: the wrong number of arguments, a `target` that is not a `geometry.Feature`, an `edges` value that is not a `geometry.RegionOutput` with edge topology, or a distance that cannot be converted, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "bevel-bore",
        "Chamfer the edges a hole introduced",
        "plate = geometry.box(80e-3, 50e-3, 8e-3);\nholed = geometry.hole(plate, 4e-3, 8e-3, [0 0 1], [40e-3 25e-3 0]);\nbeveled = geometry.chamfer(holed, geometry.edgesIntroducedBy(holed), 0.5e-3);\nclass(beveled)",
        "assert(isa(beveled, 'geometry.Feature'));",
    ),
    error_example(
        "zero-distance",
        "A zero distance is rejected",
        "block = geometry.box(40e-3, 40e-3, 20e-3);\nbeveled = geometry.chamfer(block, geometry.edges(block), 0);",
        "RunMat:geometry:chamfer:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Can the two setbacks differ?",
        answer: "No. `geometry.chamfer` applies one symmetric distance to every selected edge.",
    },
    UNITS_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.chamfer",
    summary: "Bevel selected edges of a geometry feature by a constant distance.",
    description: "`feature = geometry.chamfer(target, edges, distance)` returns a `geometry.Feature` for `target` with every edge in the typed selection `edges` replaced by a flat bevel set back `distance` meters on both adjacent faces.",
    keywords: &["geometry", "cad", "chamfer", "bevel", "edges", "feature"],
    related: &["geometry.fillet", "geometry.edges", "geometry.edgesIntroducedBy", "geometry.facesIntroducedBy"],
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
    "geometry.chamfer",
    DOCUMENTATION,
    &GEOMETRY_CHAMFER_DESCRIPTOR,
    infer,
);
