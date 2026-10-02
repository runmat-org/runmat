use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    descriptors::FEA_RUN_OPTIONS_DESCRIPTOR,
    inference::{infer_constructor, FeaConstructorKind},
    integer::FEA_RUN_OPTIONS_INTEGER_CAPABILITIES,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::RunOptions)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "transient-count",
    title: "Configure exact transient step count",
    program: "opts = fea.runOptions('transient', 'StepCount', uint32(20), 'TimeStepS', 5e-4);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(opts), 'fea.RunOptions'));",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.runOptions", summary: "Create family-specific typed FEA run options.",
    description: "`fea.runOptions(solver, Name, Value, ...)` creates a `fea.RunOptions` object whose schema is selected by the solver family.",
    keywords: &["fea", "run options", "solver", "iterations", "tolerance"], related: &["fea.study", "fea.run", "fea.step"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Solver families are linear_static, modal, acoustic, thermal, transient, cfd, cht, fsi, nonlinear, and electromagnetic.",
            "Count and iteration fields are nonnegative exact structural scalars; all eight integer classes and integral doubles are accepted exactly. Fractional, nonfinite, negative, out-of-range, nonscalar, and resident controls reject. Solver-specific positivity is checked before execution.",
            "Tolerance, time, scale, ratio, exponent, frequency, and residual fields are finite real physical scalars. Integers cross once into binary64 after solver, field, and shape validation.",
            "Logical controls require logical scalars. PrecisionMode, PreconditionerMode, QualityPolicy, PrepArtifactId, and PrepCalibrationProfile are text. Public PrepContext injection is rejected.",
            "Unknown or duplicate normalized options reject before construction. Structural properties stay exact typed integers; floating options stay double.",
        ] },
        BuiltinDocumentationSection { heading: "Exact and floating fields", paragraphs: &[
            "Modal and acoustic: exact ModeCount (default 3), floating ResidualWarnThreshold (default 1e-3). Thermal: exact StepCount (10), floating TimeStepS (1e-2) and ResidualWarnThreshold (1e-4).",
            "Transient: exact StepCount (10), MaxLinearIters (128), MaxStepRetries (4); floating TimeStepS (1e-3), MinTimeStepS (1e-6), MaxTimeStepS (2e-2), Tolerance (1e-8), ResidualTarget (1e-6), AdaptMinScale (0.8), AdaptMaxScale (1.25), AdaptGrowthExponent (0.35), AdaptRetryGrowthCap (1.05), AdaptNonconvergedShrink (0.75), DtBucketRelTolerance (0).",
            "CFD, CHT, and FSI: exact StepCount (12), MaxLinearIters (128); floating TimeStepS (1e-3), Tolerance (1e-8), and ResidualWarnThreshold (1e-5 for CFD, 1e-4 for CHT/FSI).",
            "Nonlinear: exact IncrementCount (12), MaxNewtonIters (24), MaxLineSearchBacktracks (6), TangentRefreshInterval (2); floating Tolerance (1e-6), ResidualConvergenceFactor (5), IncrementNormTolerance (1e-7), LineSearchReduction (0.5).",
            "Electromagnetic: exact HarmonicMaxIterations (96); floating ResidualTarget (1e-6), HarmonicTolerance (1e-7), and finite SweepFrequencyHz values.",
        ] },
        BuiltinDocumentationSection { heading: "Defaults", paragraphs: &[
            "All families default DeterministicMode=false, PrecisionMode=fp64, and QualityPolicy=balanced. PrepArtifactId and PrepCalibrationProfile default absent.",
            "linear_static defaults PreconditionerMode=auto and has no numeric count or physical scalar option; transient defaults AdaptiveTimeStep=true; nonlinear defaults LineSearch=true; electromagnetic defaults SweepEnabled=false and empty SweepFrequencyHz.",
        ] },
    ], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Why is an integer accepted for a tolerance?", answer: "It is a finite real scalar converted once to the option's binary64 physics field; count fields instead remain exact integers." }],
    links: &[BuiltinDocumentationLink { label: "Using FEA", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/run_options.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Run option integer boundaries", location: "crates/runmat-runtime/src/builtins/fea/tests/numeric_integrity.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "FEA VM execution", location: "crates/runmat-vm/tests/integer_builtin_fea_execution_and_feedback_file_status.rs" }],
    notes: &["Distributed option values remain unsupported until solver/field/shape preflight can run before top-level materialization."],
});
pub const FEA_RUN_OPTIONS_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.runOptions",
        documentation: DOCUMENTATION,
        descriptor: &FEA_RUN_OPTIONS_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_RUN_OPTIONS_INTEGER_CAPABILITIES,
        integer_audit: None,
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_RUN_OPTIONS_CATALOG_ENTRY];
