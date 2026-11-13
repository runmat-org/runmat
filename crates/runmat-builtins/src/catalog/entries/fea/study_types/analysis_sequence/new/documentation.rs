use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Exactly five arguments are required. `id` is non-empty text. `steps` is one family step object (for example `fea.StructuralStep`, usually configured with `fea.ProcedureStep.configure`) or a cell array of them; `dependencies`, `transitions`, and `branches` are a single typed object or a cell array, and `cell(0, 0)` declares none.",
        "Steps run in the order the dependencies imply, not the order they are listed. Model validation requires at least one step, unique step ids, an acyclic dependency graph, valid branches and transitions, and per-step checks of activations, state inputs and outputs, observations, accuracy, and restart.",
        "The sequence is immutable and portable: it contains no host paths, device handles, or worker ids, so the same sequence runs locally, on a GPU, or on a cluster with unchanged meaning.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the sequence to `fea.model(id, geometry, ..., \"Sequence\", sequence)`. `Sequence` and `Steps` are mutually exclusive: use `Steps` for independent single-step models and `Sequence` when activation, state transfer, branching, or restart matters. `fea.ExecutionPlan.defaults(sequence, backend, requirement)` binds the sequence to execution options.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the argument count is not five, `id` is empty, a step is not a supported family step, or an edge list contains the wrong class.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "static",
        title: "One-step static sequence",
        program: concat!(
            "state = fea.StateOutput.new(\"static_equilibrium\", fea.StateKind.structuralPrimary(), ...\n",
            "    fea.StateCompleteness.complete(), fea.StatePersistence.canonical());\n",
            "fixed = fea.StepObjectActivation.new(fea.StepObjectRef.constraint(\"fixed_base\"), fea.ActivationLifecycle.activate());\n",
            "storage = fea.ObservationStorage.new( ...\n",
            "    fea.SignalPrecision.real64(), fea.CompressionPolicy.lossless(), ...\n",
            "    fea.RetentionPolicy.canonical(), fea.PartialAccessPolicy.manifestAndChunks());\n",
            "cost = fea.ObservationCost.new(25000, 400000, 24000, 24000, 0);\n",
            "displacement = fea.Observation.field(\"displacement\", fea.ObservationQuantity.displacement(), ...\n",
            "    fea.ObservationTarget.wholeModel(), fea.FieldLocation.node(), fea.FieldLocation.node(), ...\n",
            "    fea.RecoveryPolicy.native(), fea.ObservationComponent.all(), fea.CoordinateFrame.globalFrame(), ...\n",
            "    fea.SignConvention.native(), fea.ComplexRepresentation.real(), fea.SamplingPolicy.finalSample(), ...\n",
            "    fea.ReductionPolicy.none(), storage, fea.ObservationPurpose.visualization(), cost);\n",
            "staticStep = fea.ProcedureStep.configure(fea.StructuralStep.static(\"static\"), ...\n",
            "    fea.IndependentDomain.staticDomain(), {fixed}, cell(0, 0), {state}, {displacement}, ...\n",
            "    fea.AccuracyPlan.baseline(\"static_acceptance\"), fea.RestartPolicy.fresh());\n",
            "sequence = fea.AnalysisSequence.new(\"bracket_sequence\", {staticStep}, cell(0, 0), cell(0, 0), cell(0, 0));",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(sequence), 'fea.AnalysisSequence'));\nassert(strcmp(sequence.sequence_id, 'bracket_sequence'));\nassert(numel(sequence.steps) == 1);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.AnalysisSequence.new",
        summary: "Assemble procedure steps, dependencies, state transitions, and branches into one analysis sequence.",
        description: "`fea.AnalysisSequence.new(id, steps, dependencies, transitions, branches)` builds the `fea.AnalysisSequence` that a model runs: its steps plus the typed edges that order them and move state between them.",
        keywords: &["fea", "analysis", "sequence", "steps", "state", "workflow"],
        related: &["fea.ProcedureStep.configure", "fea.SequenceDependency.new", "fea.StateTransition.new", "fea.SequenceBranch.new", "fea.model"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Does list order define execution order?", answer: "No. Only dependencies order steps; independent steps may run concurrently." }, BuiltinDocumentationFaq { question: "Can I pass a single step without a cell array?", answer: "Yes. Each list argument accepts one typed object or a cell array." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
            BuiltinDocumentationLink { label: "FEA models", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/models") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/analysis_sequence/tests.rs::analysis_sequence_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed sequence round trip", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Compiled sequence execution", location: "crates/runmat-vm/tests/fea_study_contracts.rs::typed_study_sequence_compiles_and_executes_through_the_vm" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Dependency cycles", location: "crates/runmat-analysis/core/src/validate_sequence/tests.rs::rejects_dependency_cycles" },
        ],
        notes: &[],
    });
