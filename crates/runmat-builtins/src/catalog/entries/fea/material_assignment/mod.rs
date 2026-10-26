use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_MATERIAL_ASSIGNMENT_DESCRIPTOR,
    doc_links,
    example_fixtures::{BOX_STUDY, NATIVE_CAD},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::MaterialAssignment)
}

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "assign-steel-to-solid",
        title: "Assign steel to the solid body of a part",
        program: concat!(
            "part = geometry.load(\"box.brep\", \"Units\", \"m\");\n",
            "steel = fea.material(\"steel\", \"YoungsModulusPa\", 200e9, \"PoissonRatio\", 0.3, \"MassDensityKgPerM3\", 7850);\n",
            "body = fea.PropertyRegionTarget.allSolids(part);\n",
            "assignment = fea.materialAssignment(body, steel);",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_STUDY,
        requirements: NATIVE_CAD,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(assignment, 'fea.MaterialAssignment'));\nassert(strcmp(assignment.assigned_material_id, \"steel\"));\nassert(strcmp(assignment.expected_material_id, \"steel\"));\nassert(strcmp(assignment.confidence, \"verified\"));\nassert(isa(assignment.geometry_revision, 'uint32'));",
        },
    },
    BuiltinExample {
        id: "record-probable-substitution",
        title: "Record a probable material substitution",
        program: concat!(
            "part = geometry.load(\"box.brep\", \"Units\", \"m\");\n",
            "steel = fea.material(\"steel\", \"YoungsModulusPa\", 200e9, \"PoissonRatio\", 0.3, \"MassDensityKgPerM3\", 7850);\n",
            "stainless = fea.material(\"stainless_304\", \"YoungsModulusPa\", 193e9, \"PoissonRatio\", 0.29, \"MassDensityKgPerM3\", 8000);\n",
            "everySolid = fea.PropertyRegionTarget.allSolids(part);\n",
            "assignment = fea.materialAssignment(everySolid, steel, \"ExpectedMaterial\", stainless, \"Confidence\", \"probable\");",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_STUDY,
        requirements: NATIVE_CAD,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(assignment.assigned_material_id, \"steel\"));\nassert(strcmp(assignment.expected_material_id, \"stainless_304\"));\nassert(strcmp(assignment.confidence, \"probable\"));",
        },
    },
];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.materialAssignment",
    summary: "Give a solid, shell, beam, or cell region of a geometry its material.",
    description: "`assignment = fea.materialAssignment(target, material)` binds a `fea.Material` to a `fea.PropertyRegionTarget`. `fea.materialAssignment(target, material, \"ExpectedMaterial\", expected, \"Confidence\", level)` also records which material the source data called for and how certain the assignment is.",
    keywords: &["fea", "material", "assignment", "region", "solid", "property"],
    related: &["fea.material", "fea.PropertyRegionTarget.solid", "fea.PropertyRegionTarget.allSolids", "fea.PropertyRegionTarget.shell", "fea.PropertyRegionTarget.beam", "geometry.region", "fea.model"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The target says which material-bearing domain receives the material: a selected solid, shell, beam, or cell region (`fea.PropertyRegionTarget.solid(region)` and its siblings), or every domain of one kind in the geometry (`fea.PropertyRegionTarget.allSolids(geometry)` and its siblings). A boundary face is not a material domain; use a shell target for a face that carries a shell section.",
            "The material must be a `fea.Material` object. The assignment stores the material id; the material definition itself goes in the model's `Materials` option. Text region selectors and material ids are not accepted.",
            "`ExpectedMaterial` (or `ExpectedMaterialId`) takes another `fea.Material` and records the material the source data specified when it differs from the one assigned, for example a documented substitution. It defaults to the assigned material.",
            "`Confidence` is `\"verified\"` (the default), `\"probable\"`, or `\"inferred\"`, matched case-insensitively. It is evidence carried into study results; it does not change the solve.",
            "The assignment records the geometry id and revision of its target in `geometry_id` and `geometry_revision`; the revision is a `uint32`, the same type as the `geometry.Asset` revision. `fea.model` rejects an assignment whose target came from a different geometry asset.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Pass assignments to `fea.model` through `MaterialAssignments`, together with the assigned materials in `Materials`. Each assigned material id must name a material in `Materials`; the model's material ownership is validated when the model is placed in a study with `fea.study`, before any solver runs.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports fewer than two arguments, a target that is not a `fea.PropertyRegionTarget`, a material that is not a `fea.Material`, an unknown option, an incomplete Name/Value pair, or a confidence other than verified, probable, or inferred.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can I assign a material to a boundary face?", answer: "No. Bulk materials belong to solid or cell regions. A face that carries a shell section takes a shell target; a face used for loads or supports is not a material domain." },
        BuiltinDocumentationFaq { question: "Does Confidence change the solution?", answer: "No. Confidence and ExpectedMaterial are provenance recorded with the model and its results; the solver uses the assigned material." },
    ],
    links: &[doc_links::MODELS, doc_links::STUDY_CONTRACTS],
    implementation: &[doc_links::FEA_MODELING_SOURCE],
    verification: &[
        doc_links::STUDY_AND_SWEEP_CONSTRUCTION_TEST,
        doc_links::ENUM_ATTRIBUTION_TEST,
        doc_links::ASSIGNMENT_REVISION_TEST,
    ],
    notes: &[],
});

pub const FEA_MATERIAL_ASSIGNMENT_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.materialAssignment",
        documentation: DOCUMENTATION,
        descriptor: &FEA_MATERIAL_ASSIGNMENT_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_MATERIAL_ASSIGNMENT_CATALOG_ENTRY];
