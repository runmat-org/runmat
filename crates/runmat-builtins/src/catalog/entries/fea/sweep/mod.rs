use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_SWEEP_DESCRIPTOR,
    doc_links,
    example_fixtures::{fea_box_model_setup, BOX_STUDY, NATIVE_CAD},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Sweep)
}

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "two-material-sweep",
    title: "Sweep the same box over two materials",
    program: concat!(
        fea_box_model_setup!(),
        "steelStudy = fea.study(\"box_steel\", part, model);\n",
        "aluminum = fea.material(\"aluminum\", \"YoungsModulusPa\", 68.9e9, \"PoissonRatio\", 0.33, \"MassDensityKgPerM3\", 2700);\n",
        "aluminumBody = fea.materialAssignment(fea.PropertyRegionTarget.allSolids(part), aluminum);\n",
        "aluminumModel = fea.model(\"box_aluminum_model\", part, \"Materials\", {aluminum}, \"MaterialAssignments\", {aluminumBody}, ...\n",
        "    \"Constraints\", {support}, \"Signals\", {tipForce}, \"Excitations\", {tipLoad}, ...\n",
        "    \"ExcitationCases\", {service}, \"Steps\", {staticStep});\n",
        "aluminumStudy = fea.study(\"box_aluminum\", part, aluminumModel);\n",
        "sweep = fea.sweep(\"box_materials\", {steelStudy, aluminumStudy}, \"FailFast\", false);",
    ),
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BOX_STUDY,
    requirements: NATIVE_CAD,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(sweep, 'fea.Sweep'));\nassert(strcmp(sweep.sweep_id, \"box_materials\"));\nassert(sweep.fail_fast == false);",
    },
}];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.sweep",
    summary: "Group FEA studies into one ordered sweep that validates, plans, and runs together.",
    description: "`sweep = fea.sweep(id, studies)` creates a `fea.Sweep` from one `fea.Study` or a cell array of studies. `fea.sweep(id, studies, \"FailFast\", false)` keeps running the remaining studies after one fails.",
    keywords: &["fea", "sweep", "parametric", "design study", "batch", "studies"],
    related: &["fea.study", "fea.validate", "fea.plan", "fea.run", "fea.results"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "A sweep is an ordered list of complete studies under one sweep id. Study order is preserved: validation entries, plan entries, run entries, and failure indexes all follow the order of the `studies` argument, never worker completion order.",
            "Each study must be a `fea.Study`, either passed alone or as elements of a cell array. Every study id must be unique within the sweep; repeating a study (or two studies with the same id) is rejected.",
            "`FailFast` is a logical scalar and defaults to `true`, which stops the sweep at the first failed study. With `false`, RunMat records the failure and continues with the remaining studies. Failure indexes reported by `fea.plan` and `fea.run` are one-based positions in the `studies` list.",
            "Construction is a pure, in-memory operation. It does not mesh, validate, or solve; those happen in `fea.validate`, `fea.plan`, and `fea.run`. Generated input signals bound to each study travel with the sweep so a later run materializes exactly the inputs each study was built with.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Build each case with `fea.study`, then pass the sweep to `fea.validate` (returns `fea.SweepValidation`), `fea.plan` (returns `fea.SweepPlan`), or `fea.run` (returns `fea.SweepRunResult`). Each successful run entry carries its own run id for `fea.results` and `fea.compare`.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports a non-text id, a value that is not a `fea.Study`, a duplicated study id, a non-logical `FailFast`, or any option other than `FailFast`.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does FailFast change?", answer: "With the default `true`, the sweep stops at the first failed study. With `false`, every study is attempted and failures are collected with their one-based study index." },
        BuiltinDocumentationFaq { question: "Can a sweep mix different geometries or physics?", answer: "Yes. Each element is a complete study with its own geometry, model, and execution plan; only the study ids must be unique." },
    ],
    links: &[doc_links::SWEEPS, doc_links::STUDY_CONTRACTS, doc_links::GET_STARTED],
    implementation: &[doc_links::FEA_MODELING_SOURCE],
    verification: &[
        doc_links::STUDY_AND_SWEEP_CONSTRUCTION_TEST,
        doc_links::SWEEP_CLASSES_TEST,
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Sweep failures cross to one-based public indexes", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_sweep_failures_cross_to_one_based_public_indices" },
    ],
    notes: &[],
});

pub const FEA_SWEEP_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.sweep",
        documentation: DOCUMENTATION,
        descriptor: &FEA_SWEEP_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_SWEEP_CATALOG_ENTRY];
