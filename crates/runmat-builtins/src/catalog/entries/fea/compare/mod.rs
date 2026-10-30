use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_COMPARE_DESCRIPTOR,
    doc_links,
    example_fixtures::{fea_box_run_setup, BOX_STUDY, NATIVE_CAD_MESHING},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    result_inference::{infer_result, FeaResultKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_result(request, entry, FeaResultKind::Compare)
}

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "compare-repeat-runs",
    title: "Compare two runs of the same study",
    program: concat!(
        fea_box_run_setup!(),
        "repeat = fea.run(study);\n",
        "comparison = fea.compare(run.steps(1).run_id, repeat.steps(1).run_id);",
    ),
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BOX_STUDY,
    requirements: NATIVE_CAD_MESHING,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(comparison, 'fea.Compare'));\nassert(~comparison.run_status_changed);\nassert(~comparison.numerically_acceptable_changed);\nassert(comparison.quality_reason_count_delta == 0);",
    },
}];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.compare",
    summary: "Compare the status, quality, and solve time of two persisted FEA runs.",
    description: "`comparison = fea.compare(baselineRunId, candidateRunId)` loads two step runs by run id and returns a `fea.Compare` describing what changed from the baseline to the candidate.",
    keywords: &["fea", "compare", "regression", "run id", "quality", "solve time"],
    related: &["fea.run", "fea.results", "fea.trends", "fea.stepResult"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Both arguments are run id strings of solved procedure steps, typically `run.steps(k).run_id` or `stepRun.run_id`. The runs are read from the analysis artifact store.",
            "The `fea.Compare` result reports `baseline_run_id` and `candidate_run_id`; logical flags `run_status_changed` and `numerically_acceptable_changed`; `quality_reason_count_delta` (candidate minus baseline); and `solve_ms_delta` in milliseconds when both runs recorded a solve time.",
            "For nonlinear runs it also reports `failed_increment_delta`, `max_iteration_delta`, `nonlinear_spike_count_delta`, and `nonlinear_stall_count_delta`. These deltas are empty when either run has no nonlinear results. Count deltas are exact signed integers.",
            "Comparison only reads stored runs; it does not re-solve or modify either run.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Use it to check a model change, a mesh refinement, or a new release against a reference run: keep the baseline step's `run_id`, run the changed study, and compare the candidate step's `run_id` against it. Use `fea.trends` to summarize many runs instead of two.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports anything other than two text run ids. `RunMat:fea:OperationFailed` reports a run id that is not in the artifact store or a run artifact that cannot be read.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can I pass fea.RunResult objects instead of run ids?", answer: "No. Pass run id strings, for example `run.steps(1).run_id`; a multi-step run has one run id per step." },
        BuiltinDocumentationFaq { question: "Does fea.compare compare field values?", answer: "No. It compares run status, quality reasons, nonlinear convergence counts, and solve time. Compare field values with `fea.field` and ordinary RunMat arithmetic." },
    ],
    links: &[doc_links::EVIDENCE, doc_links::OPERATION_REFERENCE],
    implementation: &[doc_links::FEA_ENTRYPOINTS_SOURCE],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Comparison reports typed deltas", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_results_compare_reports_typed_deltas_rejects_unsupported_envelope" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Plan counts and compare deltas stay exact", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_structural_serializer_preserves_plan_counts_and_compare_deltas" },
    ],
    notes: &[],
});

pub const FEA_COMPARE_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.compare",
        documentation: DOCUMENTATION,
        descriptor: &FEA_COMPARE_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::ArtifactRead,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_COMPARE_CATALOG_ENTRY];
