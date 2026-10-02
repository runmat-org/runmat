use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    descriptors::FEA_MODEL_DESCRIPTOR,
    inference::{infer_constructor, FeaConstructorKind},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Model)
}

const GEOMETRY: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "triangle-stl" },
    root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[BuiltinFilesystemEntry::File { relative_path: "part.stl", content: BuiltinFixtureContent::Utf8("solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n") }],
});
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "explicit-model", title: "Assemble an explicit model from geometry", program: "geom = geometry.load('part.stl');\nsteel = fea.material('steel', 'YoungsModulusPa', int64(200000000000), 'PoissonRatio', 0.3);\nmodel = fea.model('bracket', geom, 'Profile', 'linear_static_structural', 'Materials', {steel});",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem, fixture: GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(model), 'fea.Model'));" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.model", summary: "Assemble a typed RunMat FEA model.",
    description: "`fea.model(id, geometry, Name, Value, ...)` assembles an analysis model from a `geometry.Asset` and typed RunMat FEA components.",
    keywords: &["fea", "model", "geometry", "components"], related: &["geometry.load", "fea.material", "fea.loadCase"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Model id and component identities remain exact structural values; the geometry revision is copied exactly into the model.",
        "Profile, frame, defaults mode, materials, assignments, boundaries, loads, steps, domains, and interfaces are validated during assembly.",
        "Numeric physics values must already cross their owning constructor's explicit binary64 boundary; model assembly never reinterprets integer arrays as components.",
        "An existing geometry asset is required; CAD import is a separate operation. Model assembly does not intrinsically require OCCT.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Does `fea.model` import CAD?", answer: "No. Pass a `geometry.Asset`. CAD import is separate; only some CAD formats require an OCCT-enabled importer." }],
    links: &[BuiltinDocumentationLink { label: "FEA models", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/models") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/model/constructors.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "FEA model constructor", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs::fea_model_requires_profile" }],
    notes: &["`geometry.Asset` and typed component identities are local model objects; execution-owned distributed handles are not accepted as model constructor operands."],
});
pub const FEA_MODEL_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.model",
        documentation: DOCUMENTATION,
        descriptor: &FEA_MODEL_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_MODEL_CATALOG_ENTRY];
