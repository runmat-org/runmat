use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_STUDY_DESCRIPTOR,
    doc_links,
    example_fixtures::{fea_box_model_setup, BOX_STUDY, NATIVE_CAD, NATIVE_CAD_MESHING},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Study)
}

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "box-study",
        title: "Build a linear-static study for a steel box",
        program: concat!(
            fea_box_model_setup!(),
            "study = fea.study(\"box_static\", part, model);",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_STUDY,
        requirements: NATIVE_CAD,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(study, 'fea.Study'));\nassert(strcmp(study.study_id, \"box_static\"));",
        },
    },
    BuiltinExample {
        id: "meshed-study",
        title: "Attach a generated solver mesh",
        program: concat!(
            fea_box_model_setup!(),
            "solverMesh = meshing.generate(part, \"TargetSize\", 0.5, \"MaximumDeviation\", 0.05, \"ElementOrder\", \"tet4\");\n",
            "study = fea.study(\"box_static\", part, model, \"Mesh\", solverMesh);",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_STUDY,
        requirements: NATIVE_CAD_MESHING,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(study, 'fea.Study'));\nassert(~isempty(study.solver_mesh_artifact_path));",
        },
    },
];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.study",
    summary: "Bind a geometry, a model, and execution settings into a typed FEA study.",
    description: "`study = fea.study(id, geometry, model)` creates a `fea.Study` from a `geometry.Asset` and a `fea.Model` built on that same geometry revision. `fea.study(id, geometry, model, \"Mesh\", mesh, \"Execution\", plan)` also attaches a validated solver mesh and an explicit execution plan.",
    keywords: &["fea", "study", "simulation", "geometry", "model", "mesh", "execution"],
    related: &["fea.model", "fea.validate", "fea.plan", "fea.run", "fea.sweep", "geometry.load", "meshing.generate"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "A study is the unit RunMat validates, plans, and solves. It holds the study id, the exact `geometry.Asset`, the `fea.Model`, an execution plan, and (when supplied) the persisted solver mesh and its meshing evidence. The returned `fea.Study` is an immutable value: changing any input means building a new study.",
            "The id must be text. The geometry must be a `geometry.Asset` and the model a `fea.Model`. The model must be bound to the same geometry id and revision as the geometry argument; a model built on another asset, or on an older revision of the same file, is rejected.",
            "The model is fully validated before the study is returned. Missing materials, a model without any constraint or boundary, invalid component references, an inconsistent procedure sequence, or a model whose units or coordinate frame disagree with the geometry all fail at construction with `RunMat:fea:InvalidInput`.",
            "`Execution` takes a `fea.ExecutionPlan` (for example from `fea.ExecutionPlan.defaults`). The plan must cover every procedure step in the model's sequence. Without it, RunMat derives the default plan for the sequence on the CPU backend.",
            "`Mesh` takes a `meshing.Mesh` from `meshing.generate` or `meshing.load`. Region aliases recorded by the mesher are resolved onto the model's region references, and the mesh and its evidence are persisted to the configured analysis artifact store so that `fea.run` can solve on exactly that mesh. A study without a mesh can be validated and planned, but solving it requires a canonical solver mesh.",
            "Option names are matched case-insensitively with punctuation ignored. Each option may appear once; any other option name is an error.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Build the inputs with `geometry.load`, `fea.material`, `fea.materialAssignment`, constraint and excitation constructors, a step constructor such as `fea.StructuralStep.static`, and `fea.model`. Pass the study to `fea.validate` to check it, `fea.plan` to see the solver operations it will run, and `fea.run` to solve it. Combine several studies with `fea.sweep`.",
            "The same study can also be authored in a `.fea` file and run from the command line with `runmat run`; the example fixture includes `box_static.fea`, the `.fea` form of the scripted study below.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports a missing or wrongly typed argument, an unsupported or repeated option, a geometry mismatch, or a model that fails validation. `RunMat:fea:OperationFailed` reports a failure to persist the supplied mesh into the artifact store. `RunMat:fea:Internal` reports a study that cannot be converted to a RunMat value.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "Do I need a mesh to create a study?", answer: "No. A study without `Mesh` can be validated and planned. To solve it with `fea.run`, attach a solver mesh from `meshing.generate` or `meshing.load`." },
        BuiltinDocumentationFaq { question: "Which backend runs the study by default?", answer: "Without an `Execution` plan, every procedure step is planned for the CPU backend. Pass a `fea.ExecutionPlan` to request a different backend or formulation maturity." },
        BuiltinDocumentationFaq { question: "Can I reuse a model with a different geometry file?", answer: "No. A model records the geometry id and revision it was built on. Rebuild the model with `fea.model` on the new `geometry.Asset`." },
    ],
    links: &[doc_links::USING_FEA, doc_links::STUDY_CONTRACTS, doc_links::MESHING],
    implementation: &[doc_links::FEA_MODELING_SOURCE, doc_links::FEA_ENTRYPOINTS_SOURCE],
    verification: &[
        doc_links::STUDY_AND_SWEEP_CONSTRUCTION_TEST,
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "A study requires a geometry asset", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_study_requires_geometry_asset" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "A study requires an explicit model", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_study_requires_explicit_model" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "The Mesh option accepts a jointly validated meshing object", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_study_options_accept_a_jointly_validated_meshing_object" },
    ],
    notes: &["Supplying `Mesh` persists the mesh and its evidence into the analysis artifact store; that write is the only external effect of this constructor."],
});

pub const FEA_STUDY_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::artifact_writing_entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.study",
        documentation: DOCUMENTATION,
        descriptor: &FEA_STUDY_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_STUDY_CATALOG_ENTRY];
