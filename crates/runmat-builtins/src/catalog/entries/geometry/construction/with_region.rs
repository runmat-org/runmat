//! `geometry.withRegion` catalog identity.

use super::inference::{
    infer_geometry, GeometryArgument as A, GeometrySignature, GENERATED_RESULT,
};
use super::*;
use runmat_types::{CallInference, CallRequest};

const SIGNATURE: GeometrySignature = GeometrySignature {
    code: "WITH_REGION",
    name: "geometry.withRegion",
    positional: &[A::Generated, A::Text, A::RegionOutput],
    variadic: None,
    result: GENERATED_RESULT,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_geometry(request, entry, SIGNATURE)
}

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`name` must be nonblank, at most 256 bytes, free of control characters, and not already published on the part. `output` comes from `geometry.solid`, `geometry.faces`, `geometry.edges`, `geometry.facesIntroducedBy`, or `geometry.edgesIntroducedBy`, and it must name a feature in the part's graph. The input part is unchanged; assign the result to keep the region.",
            "The role (solid, face, or edge) and cardinality (exactly one or one or more) travel with the output, so the `.fea` region that references it is checked against them when the part is realized.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            USE_IN_A_STUDY,
            "Each published name becomes `output.region.<name>` in the `.fea` geometry declaration, for example `region bore : face { select output.region.bore; expect one_or_more }`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:geometry:withRegion:InvalidInput` reports the wrong number of arguments, a part that is not `geometry.Generated`, an output that is not `geometry.RegionOutput` or that names a feature outside the part's graph, a name that is not scalar text or is invalid, or a name that is already published.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    example(
        "publish-regions",
        "Publish body and bore regions",
        "stock = geometry.box(80e-3, 50e-3, 8e-3);\nbody = geometry.hole(stock, 5e-3, 8e-3, [0 0 1], [40e-3 25e-3 0]);\npart = geometry.finish(body);\npart = geometry.withRegion(part, \"body\", geometry.solid(body));\npart = geometry.withRegion(part, \"bore\", geometry.facesIntroducedBy(body));\nclass(part)",
        "assert(isa(part, 'geometry.Generated'));",
    ),
    error_example(
        "duplicate-name",
        "A name can be published only once",
        "block = geometry.box(40e-3, 40e-3, 20e-3);\npart = geometry.withRegion(geometry.finish(block), \"body\", geometry.solid(block));\npart = geometry.withRegion(part, \"body\", geometry.solid(block));",
        "RunMat:geometry:withRegion:InvalidInput",
    ),
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Can I rename or remove a region?",
        answer: "No. Products are immutable and names are unique; rebuild the part from `geometry.finish` with the regions you want.",
    },
    REALIZATION_FAQ,
];

const DOCUMENTATION: BuiltinDocumentation = documentation(ConstructionDocumentation {
    title: "geometry.withRegion",
    summary: "Publish a typed selection under a name on a generated-geometry part.",
    description: "`part = geometry.withRegion(part, name, output)` returns a new `geometry.Generated` product with the region output `output` published as `name`.",
    keywords: &["geometry", "cad", "region", "name", "output", "generated", "selection"],
    related: &["geometry.finish", "geometry.solid", "geometry.faces", "geometry.edges", "geometry.facesIntroducedBy", "geometry.edgesIntroducedBy"],
    sections: SECTIONS,
    examples: EXAMPLES,
    faqs: FAQS,
    implementation: &[REGIONS_SOURCE, GRAPH_VALIDATION_SOURCE],
    verification: &[COMPOSITION_TEST, FAIL_CLOSED_TEST, ERROR_OWNER_TEST],
    notes: &[],
    status: BuiltinDocumentationStatus::Stable,
});

pub(super) const ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "geometry.withRegion",
    DOCUMENTATION,
    &GEOMETRY_WITH_REGION_DESCRIPTOR,
    infer,
);
