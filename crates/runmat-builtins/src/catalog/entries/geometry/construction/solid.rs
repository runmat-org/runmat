//! `geometry.solid` catalog identity.

use super::inference::{
    infer_geometry, GeometryArgument as A, GeometrySignature, REGION_OUTPUT_RESULT,
};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "SOLID",
    name: "geometry.solid",
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
            "The selection names the feature, not a topology index. When the part is realized it resolves to the solids of `feature`'s result that are present in the finished part, and it fails unless exactly one remains. A disjoint union or an instance pattern has several solids and does not satisfy `geometry.solid`.",
            "Selections are defined by feature lineage, not by entity index or position. Because the region names what the feature produced, it keeps selecting the intended topology when dimensions change and the part is regenerated, and realization fails rather than silently selecting different topology.",
            "Creating the selection is a pure, non-suspending call; it does not realize geometry.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Publish the selection with `geometry.withRegion(part, \"body\", geometry.solid(body))`. In `.fea`, `region body : solid { select output.region.body; expect exactly_one }` then gives the model a solid region for material assignment.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:solid:InvalidInput` reports a call that does not pass exactly one `geometry.Feature` value. Selection errors at realization, such as an empty selection, a cardinality mismatch, or missing topology history, are reported by the `.fea` geometry materialization that resolves the region, before meshing or assembly.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "body-region",
        "Publish the body as a solid region",
        "stock = geometry.box(80e-3, 50e-3, 8e-3);\nbody = geometry.hole(stock, 5e-3, 8e-3, [0 0 1], [40e-3 25e-3 0]);\npart = geometry.withRegion(geometry.finish(body), \"body\", geometry.solid(body));\nclass(geometry.solid(body))",
        "assert(isa(geometry.solid(body), 'geometry.RegionOutput')); assert(isa(part, 'geometry.Generated'));",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Which feature should I pass?",
        answer: "Pass the final feature of the body, the same feature you finish. A solid selected from an intermediate feature resolves only if that solid survives unchanged in the finished part.",
    },
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.solid",
    summary: "Select the single solid produced by a geometry feature.",
    description: "`output = geometry.solid(feature)` returns a solid `geometry.RegionOutput` that selects the result solid of `feature` and requires exactly one.",
    keywords: &["geometry", "cad", "region", "solid", "body", "selection"],
    related: &["geometry.faces", "geometry.edges", "geometry.withRegion", "geometry.finish", "fea.materialAssignment"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[REGIONS_SOURCE],
    verification: &[COMPOSITION_TEST, ERROR_OWNER_TEST],
    notes: &[],
    status: BuiltinDocumentationStatus::Stable,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.solid",
    DOCUMENTATION,
    &GEOMETRY_SOLID_DESCRIPTOR,
    infer,
);
