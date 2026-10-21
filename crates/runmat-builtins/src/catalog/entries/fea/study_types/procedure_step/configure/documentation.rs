use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Exactly eight arguments are required. `step` must be a family step object: `fea.StructuralStep`, `fea.ThermalStep`, `fea.ElectromagneticStep`, `fea.AcousticStep`, `fea.FluidStep`, `fea.ThermoMechanicalStep`, `fea.ElectroThermalStep`, `fea.ConjugateHeatTransferStep`, or `fea.FluidStructureStep`. The result keeps that class, its step id, physics family, procedure, and settings.",
        "The independent domain, activations, state inputs, state outputs, observations, accuracy plan, and restart policy are replaced, not merged. List arguments accept one typed object or a cell array; `cell(0, 0)` means none. An empty activation list keeps the single-step convention that every authored object is active; a non-empty list is closed.",
        "The constructor decodes each value and fails before returning if any has the wrong class. Cross-object checks (domain versus procedure, sampling versus domain, state availability, activation lifecycles) run when the model is validated.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass configured steps to `fea.AnalysisSequence.new`, or directly to `fea.model(..., \"Steps\", {step})` for an independent single-step model.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the argument count is not eight, `step` is not a supported family step, or another argument is not the expected typed value.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "static",
        title: "Configure a static structural step",
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
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(staticStep), 'fea.StructuralStep'));\nassert(strcmp(staticStep.step_id, 'static'));\nassert(numel(staticStep.observations) == 1);\nassert(numel(staticStep.state_outputs) == 1);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ProcedureStep.configure",
        summary: "Attach a domain, activations, state, observations, accuracy, and restart to a family step.",
        description: "`fea.ProcedureStep.configure(step, independentDomain, activations, stateInputs, stateOutputs, observations, accuracy, restart)` returns a copy of a family procedure step with its study contracts replaced by the given typed values.",
        keywords: &["fea", "procedure", "step", "configure", "observation", "accuracy", "state"],
        related: &["fea.StructuralStep.static", "fea.IndependentDomain.staticDomain", "fea.StepObjectActivation.new", "fea.StateOutput.new", "fea.Observation.field", "fea.AccuracyPlan.new", "fea.RestartPolicy.fresh", "fea.AnalysisSequence.new"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Why is the result a `fea.StructuralStep` and not a `fea.ProcedureStep`?", answer: "`fea.ProcedureStep.configure` is a static method that works for every family; the configured step keeps its family class." }, BuiltinDocumentationFaq { question: "Can I configure a step twice?", answer: "Yes. Each call replaces the study contracts and returns a new immutable step." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/procedure_step/tests.rs::procedure_step_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed sequence round trip", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Compiled sequence execution", location: "crates/runmat-vm/tests/fea_study_contracts.rs::typed_study_sequence_compiles_and_executes_through_the_vm" },
        ],
        notes: &[],
    });
