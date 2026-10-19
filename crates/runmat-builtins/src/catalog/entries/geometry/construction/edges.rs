//! `geometry.edges` catalog identity.

use super::inference::{
    infer_geometry, GeometryArgument as A, GeometrySignature, REGION_OUTPUT_RESULT,
};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "EDGES",
    name: "geometry.edges",
    positional: &[A::Feature],
    variadic: None,
    result: REGION_OUTPUT_RESULT,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_geometry(request, entry, SIGNATURE)
}

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The selection resolves, when realized, to the edges of `feature`'s own result that are present in the part being built. It is the edge input to `geometry.fillet` and `geometry.chamfer`, and it can also be published as an edge region.",
            "Selections are defined by feature lineage, not by entity index or position. Because the region names what the feature produced, it keeps selecting the intended topology when dimensions change and the part is regenerated, and realization fails rather than silently selecting different topology.",
            "Creating the selection is a pure, non-suspending call; it does not realize geometry.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the selection to `geometry.fillet` or `geometry.chamfer`, or publish it with `geometry.withRegion` for an edge region in `.fea`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:edges:InvalidInput` reports a call that does not pass exactly one `geometry.Feature` value. Selection errors at realization, such as an empty selection, a cardinality mismatch, or missing topology history, are reported by the `.fea` geometry materialization that resolves the region, before meshing or assembly.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "all-edges",
        "Select every edge of a block for a chamfer",
        "block = geometry.box(40e-3, 40e-3, 20e-3);\nallEdges = geometry.edges(block);\nbeveled = geometry.chamfer(block, allEdges, 1e-3);\nclass(allEdges)",
        "assert(isa(allEdges, 'geometry.RegionOutput')); assert(isa(beveled, 'geometry.Feature'));",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why does fillet reject my selection?",
        answer: "`geometry.fillet` and `geometry.chamfer` require edge selections. `geometry.faces` and `geometry.solid` select other topology.",
    },
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.edges",
    summary: "Select the edges produced by a geometry feature.",
    description: "`output = geometry.edges(feature)` returns an edge `geometry.RegionOutput` that selects the edges of `feature`'s result and requires at least one.",
    keywords: &["geometry", "cad", "region", "edges", "selection", "fillet"],
    related: &["geometry.edgesIntroducedBy", "geometry.fillet", "geometry.chamfer", "geometry.faces", "geometry.withRegion"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[REGIONS_SOURCE],
    verification: &[SELECTION_TEST, FULL_SURFACE_TEST],
    notes: &[],
    status: BuiltinDocumentationStatus::Stable,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.edges",
    DOCUMENTATION,
    &GEOMETRY_EDGES_DESCRIPTOR,
    infer,
);
