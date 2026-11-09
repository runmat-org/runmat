use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_RUN_DESCRIPTOR,
    doc_links,
    example_fixtures::{fea_box_run_setup, BOX_STUDY, NATIVE_CAD_MESHING},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    operation_inference::{infer_evidence, FeaEvidenceKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_evidence(request, entry, FeaEvidenceKind::Run)
}

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "run-box-static",
    title: "Solve a meshed linear-static study",
    program: fea_box_run_setup!(),
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BOX_STUDY,
    requirements: NATIVE_CAD_MESHING,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(run, 'fea.RunResult'));\nassert(strcmp(run.study_id, \"box_static\"));\nassert(numel(run.steps) == 1);\nassert(~isempty(run.evidence_artifact_path));",
    },
}];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.run",
    summary: "Solve a study or sweep and record its results and evidence.",
    description: "`run = fea.run(study)` solves every procedure step of a `fea.Study` and returns a `fea.RunResult`. `fea.run(sweep)` solves each study of a `fea.Sweep` in order and returns a `fea.SweepRunResult`.",
    keywords: &["fea", "run", "solve", "simulation", "study", "sweep", "results"],
    related: &["fea.study", "fea.validate", "fea.plan", "fea.results", "fea.stepResult", "fea.field", "fea.sweep"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Running a study first materializes any generated input signals the study carries (for example signals computed by a RunMat function), then plans and executes each procedure step with the backend and solver options from the study's execution plan. Steps run in sequence order and pass state between them as the sequence declares.",
            "The study must carry a canonical solver mesh, supplied with `fea.study(..., \"Mesh\", mesh)`. A study without one fails with `RunMat:fea:OperationFailed`.",
            "The returned `fea.RunResult` holds the study and model ids, the sequence identity, one entry per step in `steps` (each with its own `run_id`, run status, quality gates, provenance, and evidence path), published states, the overall `run_status`, `numerically_acceptable` flag, solver-convergence and result-quality gates, the study fingerprint, and the run evidence path. It also keeps the study, so results, fields, plots, and mesh views can be derived from it without passing the study again.",
            "For a sweep, the `fea.SweepRunResult` reports `study_count`, `success_count`, `failed_count`, ordered `run_entries`, and `failure_entries` with one-based study indexes. With `FailFast` set to `true`, the first failing study raises an error.",
            "When RunMat runs inside a host with an analysis service (for example RunMat Desktop or a remote cluster), the solve is delegated to that service and the call suspends until it completes. Otherwise the study is solved in-process. The execution plan may select a GPU backend; host decoding and result serialization do not imply a CPU-only solve.",
            "Results, fields, and evidence are persisted in the analysis artifact store so they can be reopened later by run id.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Call `fea.run` on the object returned by `fea.study` or `fea.sweep`. For a single-step study, pass the run straight to `fea.results` or `fea.field`; for a multi-step study, select a step with `run.step(stepId)` or `fea.stepResult(run, stepId)` first. Keep the step `run_id` to reopen results later or to compare runs with `fea.compare`.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports an argument that is not a `fea.Study` or `fea.Sweep`, or a generated signal that cannot be materialized. `RunMat:fea:OperationFailed` reports an invalid study, a missing solver mesh, a solver failure, or a failure to persist results. `RunMat:fea:Internal` reports a result that cannot be converted to a RunMat value.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does fea.run always use the CPU?", answer: "No. The backend comes from the study's execution plan. Without an explicit `Execution` option, `fea.study` plans every step for the CPU." },
        BuiltinDocumentationFaq { question: "How do I reopen results in a later session?", answer: "Save the step's `run_id` and call `fea.results(runId)` in any process that can see the same artifact store." },
        BuiltinDocumentationFaq { question: "Why does my study fail with a missing mesh?", answer: "Solving needs a canonical solver mesh. Generate one with `meshing.generate(part, ...)` and pass it with `fea.study(id, part, model, \"Mesh\", mesh)`." },
    ],
    links: &[doc_links::SOLVES, doc_links::EVIDENCE, doc_links::GET_STARTED, doc_links::MESHING],
    implementation: &[doc_links::FEA_ENTRYPOINTS_SOURCE, doc_links::ANALYSIS_OPERATIONS_SOURCE],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "A study run executes the linear-static path", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_run_study_executes_linear_static_path" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "A study run preserves the requested solver mesh", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_run_study_preserves_requested_canonical_solver_mesh_artifact" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "A sweep run executes multiple studies", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_run_study_sweep_executes_multiple_studies" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Fail-fast sweep runs stop on an invalid study", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_run_study_sweep_fail_fast_returns_error_on_invalid_study" },
        doc_links::STUDY_AND_SWEEP_CONSTRUCTION_TEST,
    ],
    notes: &["A distributed array cannot stand in for a study; placement on remote or distributed capacity is selected by the host execution layer, not by this builtin."],
});

pub const FEA_RUN_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.run",
        documentation: DOCUMENTATION,
        descriptor: &FEA_RUN_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::EvidenceWrite,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_RUN_CATALOG_ENTRY];
