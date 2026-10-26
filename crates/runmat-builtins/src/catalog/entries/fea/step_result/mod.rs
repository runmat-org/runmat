use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    descriptors::FEA_STEP_RESULT_DESCRIPTOR,
    doc_links,
    example_fixtures::{fea_box_run_setup, BOX_STUDY, NATIVE_CAD_MESHING},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    result_inference::{infer_result, FeaResultKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_result(request, entry, FeaResultKind::StepResult)
}

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-static-step",
    title: "Select the static step of a run",
    program: concat!(
        fea_box_run_setup!(),
        "staticRun = fea.stepResult(run, \"static\");\n",
        "results = fea.results(staticRun);",
    ),
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BOX_STUDY,
    requirements: NATIVE_CAD_MESHING,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(staticRun, 'fea.StepRunResult'));\nassert(~isempty(staticRun.run_id));\nassert(isa(results, 'fea.Results'));",
    },
}];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.stepResult",
    summary: "Select one procedure step's result from a multi-step FEA run.",
    description: "`stepRun = fea.stepResult(run, stepId)` returns the `fea.StepRunResult` for the step named `stepId` in an aggregate `fea.RunResult`. The method form `run.step(stepId)` is equivalent.",
    keywords: &["fea", "run", "step", "sequence", "result", "multi-step"],
    related: &["fea.run", "fea.results", "fea.field", "fea.StructuralStep.static"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "A study's procedure sequence can hold several steps, for example a modal extraction followed by a response analysis. `fea.run` returns one aggregate `fea.RunResult`; each step's outcome is stored separately with its own run id. `fea.stepResult` picks one of them by its authored step id.",
            "The step id must match exactly (case-sensitive). The returned `fea.StepRunResult` holds the step's ordinal, procedure step, execution settings, run identity and operation, `run_id`, run status, `numerically_acceptable` flag, convergence and quality gates, quality reasons, provenance, and evidence path. The study carried by the aggregate run is kept on the step result.",
            "Selection is a pure, in-memory operation on the run object; it does not read the artifact store.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Use it between `fea.run` and the result functions: `fea.results(fea.stepResult(run, \"modes\"))`, `fea.field(stepRun, selector)`, or `mesh(stepRun)`. A single-step run can go straight to `fea.results` without selecting a step.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports a first argument that is not a `fea.RunResult`, a non-text step id, or a step id that the run does not contain.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "Where do step ids come from?", answer: "From the step constructors, for example `fea.StructuralStep.static(\"static\")`, or from the analysis names in a `.fea` study." },
    ],
    links: &[doc_links::STUDY_CONTRACTS, doc_links::EVIDENCE],
    implementation: &[doc_links::FEA_RUN_RESULTS_SOURCE],
    verification: &[doc_links::STEP_RESULT_TEST, doc_links::SWEEP_CLASSES_TEST],
    notes: &[],
});

pub const FEA_STEP_RESULT_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.stepResult",
        documentation: DOCUMENTATION,
        descriptor: &FEA_STEP_RESULT_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_STEP_RESULT_CATALOG_ENTRY];
