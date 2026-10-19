//! `geometry.fillet` catalog identity.

use super::inference::{infer_geometry, GeometryArgument as A, GeometrySignature, FEATURE_RESULT};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "FILLET",
    name: "geometry.fillet",
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
            "`edges` must be an edge selection from `geometry.edges` or `geometry.edgesIntroducedBy`; a face or solid selection is rejected immediately. The selection must name a feature in `target`'s graph. When the graph is realized, every selected edge must exist in the target's topology, and a radius too large for the adjacent faces fails rather than producing partial rounding.",
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
            "Fillets remove sharp re-entrant corners that concentrate stress in a linear static result. `geometry.facesIntroducedBy(rounded)` selects the new blend faces.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:fillet:InvalidInput` reports every failure of the call: the wrong number of arguments, a `target` that is not a `geometry.Feature`, an `edges` value that is not a `geometry.RegionOutput` with edge topology, or a radius that cannot be converted, or a value that violates the feature-graph contract, such as a zero or negative dimension, a non-finite value, or an all-zero direction. A failed call returns nothing, and earlier values are unchanged because builder values are immutable.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "round-block",
        "Round every edge of a block by 1 mm",
        "block = geometry.box(40e-3, 40e-3, 20e-3);\nrounded = geometry.fillet(block, geometry.edges(block), 1e-3);\nclass(rounded)",
        "assert(isa(rounded, 'geometry.Feature'));",
    ),
    error_example(
        "face-selection",
        "A face selection is rejected",
        "block = geometry.box(40e-3, 40e-3, 20e-3);\nrounded = geometry.fillet(block, geometry.faces(block), 1e-3);",
        "RunMat:geometry:fillet:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Can I fillet only the edges a cut created?",
        answer: "Yes. Pass `geometry.edgesIntroducedBy(cut)`, which selects only the edges that operation introduced.",
    },
    UNITS_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.fillet",
    summary: "Round selected edges of a geometry feature with a constant radius.",
    description: "`feature = geometry.fillet(target, edges, radius)` returns a `geometry.Feature` for `target` with every edge in the typed selection `edges` rounded to `radius` meters.",
    keywords: &["geometry", "cad", "fillet", "round", "blend", "edges", "feature"],
    related: &["geometry.chamfer", "geometry.edges", "geometry.edgesIntroducedBy", "geometry.facesIntroducedBy"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[FEATURES_SOURCE, GRAPH_VALIDATION_SOURCE],
    verification: &[FULL_SURFACE_TEST, ERROR_OWNER_TEST],
    notes: &[],
    status: BuiltinDocumentationStatus::Stable,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.fillet",
    DOCUMENTATION,
    &GEOMETRY_FILLET_DESCRIPTOR,
    infer,
);
