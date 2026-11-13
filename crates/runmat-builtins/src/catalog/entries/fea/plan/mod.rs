use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_PLAN_DESCRIPTOR,
    doc_links,
    example_fixtures::{fea_box_model_setup, BOX_STUDY, NATIVE_CAD},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    operation_inference::{infer_evidence, FeaEvidenceKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_evidence(request, entry, FeaEvidenceKind::Plan)
}

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "plan-box-study",
    title: "See which solver operation a study will run",
    program: concat!(
        fea_box_model_setup!(),
        "study = fea.study(\"box_static\", part, model);\n",
        "plan = fea.plan(study);",
    ),
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BOX_STUDY,
    requirements: NATIVE_CAD,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(plan, 'fea.Plan'));\nassert(strcmp(plan.study_id, \"box_static\"));\nassert(strcmp(plan.sequence_id, \"box_model.sequence\"));\nassert(numel(plan.steps) == 1);\nassert(numel(plan.operation_sequence) >= 1);\nassert(~isempty(plan.study_fingerprint));",
    },
}];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.plan",
    summary: "Resolve a study or sweep into the ordered solver operations it will run, without solving.",
    description: "`plan = fea.plan(study)` returns a `fea.Plan` listing each procedure step with its execution settings and the solver operation selected for it. `fea.plan(sweep)` returns a `fea.SweepPlan` with one plan entry per study.",
    keywords: &["fea", "plan", "solver", "operation sequence", "study", "sweep", "fingerprint"],
    related: &["fea.validate", "fea.run", "fea.study", "fea.sweep"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Planning first applies every `fea.validate` rule. A study with any validation issue fails with `RunMat:fea:OperationFailed` and the message points to `fea.validate` for details; the issue codes are attached to the error.",
            "For a valid study, each step of the model's procedure sequence is paired with its execution entry (backend, solver options, formulation requirement) and mapped to a supported solver operation such as the linear-static structural solve. The `fea.Plan` holds the study, model, and sequence ids, the ordered `steps`, the `operation_sequence`, a `study_fingerprint` digest of the study content, and the path of a content-addressed plan evidence artifact.",
            "For a sweep, the `fea.SweepPlan` reports `study_count`, `planned_count`, and `failed_count`, the ordered `plan_entries`, and `failure_entries` whose study indexes are one-based. With the sweep's `FailFast` set to `true`, the first failing study raises an error instead.",
            "Planning does not mesh or solve, and it does not suspend. The input must be a `fea.Study` or `fea.Sweep` object; `.fea` paths are not accepted.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Run `fea.plan` after `fea.validate` to confirm the backend and solver operation before committing to `fea.run`. The study fingerprint in the plan matches the one recorded by a later run of the same study.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports an argument that is not a `fea.Study` or `fea.Sweep`. `RunMat:fea:OperationFailed` reports an invalid study, an execution plan that does not match the sequence, or a failure to persist plan evidence.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "What is the study fingerprint?", answer: "A canonical digest of the study's content: geometry, model, execution plan, and inputs. Planning the same study twice yields the same fingerprint, and the run of that study records it too." },
        BuiltinDocumentationFaq { question: "Does planning require a mesh?", answer: "No. Planning resolves solver operations from the model and execution plan. Attach a mesh before `fea.run`." },
    ],
    links: &[doc_links::SOLVES, doc_links::STUDY_CONTRACTS, doc_links::OPERATION_REFERENCE],
    implementation: &[doc_links::FEA_ENTRYPOINTS_SOURCE, doc_links::ANALYSIS_OPERATIONS_SOURCE],
    verification: &[
        doc_links::PORTABLE_SCHEMA_TEST,
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "A linear-static study plans its canonical sequence", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_plan_study_returns_canonical_linear_static_sequence" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Sweep planning returns typed plan entries", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_plan_study_sweep_returns_typed_plan_entries" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Sweep planning continues after a study failure", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_plan_study_sweep_can_continue_on_study_failure" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Plan counts and compare deltas stay exact", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_structural_serializer_preserves_plan_counts_and_compare_deltas" },
    ],
    notes: &[],
});

pub const FEA_PLAN_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.plan",
        documentation: DOCUMENTATION,
        descriptor: &FEA_PLAN_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::SynchronousEvidenceWrite,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_PLAN_CATALOG_ENTRY];
