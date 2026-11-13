//! `geometry.faces` catalog identity.

use super::inference::{
    infer_geometry, GeometryArgument as A, GeometrySignature, REGION_OUTPUT_RESULT,
};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "FACES",
    name: "geometry.faces",
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
            "When the part is realized the selection resolves to the faces of `feature`'s own result that are present in the finished part; it fails if none remain. Use `geometry.facesIntroducedBy` instead to select only the faces an operation created, such as a bore wall.",
            "Selections are defined by feature lineage, not by entity index or position. Because the region names what the feature produced, it keeps selecting the intended topology when dimensions change and the part is regenerated, and realization fails rather than silently selecting different topology.",
            "Creating the selection is a pure, non-suspending call; it does not realize geometry.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Publish the selection with `geometry.withRegion`, then reference it in `.fea` as a face region, for example `region skin : face { select output.region.skin; expect one_or_more }`, for convection, pressure, or traction boundaries.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:faces:InvalidInput` reports a call that does not pass exactly one `geometry.Feature` value. Selection errors at realization, such as an empty selection, a cardinality mismatch, or missing topology history, are reported by the `.fea` geometry materialization that resolves the region, before meshing or assembly.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "skin",
        "Publish every face of a block",
        "block = geometry.box(40e-3, 40e-3, 20e-3);\nskin = geometry.faces(block);\npart = geometry.withRegion(geometry.finish(block), \"skin\", skin);\nclass(skin)",
        "assert(isa(skin, 'geometry.RegionOutput')); assert(isa(part, 'geometry.Generated'));",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Can I pick a single face of a box by index?",
        answer: "No. Selections follow feature lineage, not indices. Build the face with the feature that should own it and select it with `geometry.facesIntroducedBy`.",
    },
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.faces",
    summary: "Select the faces produced by a geometry feature.",
    description: "`output = geometry.faces(feature)` returns a face `geometry.RegionOutput` that selects the faces of `feature`'s result and requires at least one.",
    keywords: &["geometry", "cad", "region", "faces", "surface", "selection"],
    related: &["geometry.facesIntroducedBy", "geometry.edges", "geometry.solid", "geometry.withRegion"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[REGIONS_SOURCE],
    verification: &[SELECTION_TEST],
    notes: &[],
    status: BuiltinDocumentationStatus::Stable,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.faces",
    DOCUMENTATION,
    &GEOMETRY_FACES_DESCRIPTOR,
    infer,
);
