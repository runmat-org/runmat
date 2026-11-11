use super::header::{self, AnalysisHeader, ProcedureHeader};
use super::{requirements, values};
use crate::{
    CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind,
    FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase,
};
use runmat_analysis_core::{
    validate_analysis_sequence, AnalysisProcedure, AnalysisSequence, AuthoredModel,
    CoupledPhysicsFamily, IndependentDomain, InitialStateConflictPolicy, PhysicsFamily,
    ProcedureStep, ProcedureStepId, RandomExcitationId, ReferenceStatePolicy, ResponseSpectrumId,
    SequenceDependency, SequenceDependencyReason, SolutionSpace, StateCompleteness, StateId,
    StateInput, StateKind, StateOutput, StatePersistence, StateSource, StateTransition,
    StateTransitionMethod, StateUse, StructuralKinematics, StructuralNonlinearSettings,
    StructuralRandomResponseSettings, TopologyTransitionPolicy, TransitionId,
};
use std::collections::BTreeSet;

pub fn lower_study(
    declaration: &CheckedFeaDeclaration,
    models: &[&AuthoredModel],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AnalysisSequence> {
    let initial_errors = diagnostics.len();
    let mut analyses = Vec::new();
    for statement in &declaration.statements {
        let CheckedFeaStatementKind::Block {
            header: source,
            header_span,
            statements,
        } = &statement.kind
        else {
            continue;
        };
        match header::analysis(source, *header_span) {
            Ok(Some(header)) => analyses.push((header, statements.as_slice())),
            Ok(None) => {}
            Err(message) => error(declaration, *header_span, message, diagnostics),
        }
    }
    if analyses.is_empty() {
        error(
            declaration,
            declaration.span,
            "study requires at least one `analysis` declaration",
            diagnostics,
        );
        return None;
    }

    let mut names = BTreeSet::new();
    let mut steps = Vec::new();
    let mut dependencies = Vec::new();
    let mut transitions = Vec::new();
    for (analysis, statements) in analyses {
        if !names.insert(analysis.name.clone()) {
            error(
                declaration,
                analysis.span,
                format!("duplicate analysis `{}`", analysis.name),
                diagnostics,
            );
            continue;
        }
        let Some(procedure) = procedure_block(declaration, statements, diagnostics) else {
            continue;
        };
        let Some(selection) =
            super::load_cases::lower(declaration, procedure.1, models, diagnostics)
        else {
            continue;
        };
        let procedure_statements = procedure
            .1
            .iter()
            .filter(|statement| !super::load_cases::is_selection(statement))
            .cloned()
            .collect::<Vec<_>>();
        let (lowered, read) = values::track_members(|| {
            lower_procedure(
                declaration,
                &analysis,
                &procedure.0,
                &procedure_statements,
                models,
                diagnostics,
            )
        });
        let Some(mut step) = lowered else {
            continue;
        };
        reject_unknown_members(
            declaration,
            &procedure.0,
            &procedure_statements,
            &read,
            diagnostics,
        );
        step.excitation_case_selection = selection;
        if !super::accuracy::apply(declaration, statements, &mut step, diagnostics) {
            continue;
        }
        if let Some(retained) = values::retained_name(statements) {
            if step.state_outputs.len() == 1 {
                step.state_outputs[0].state_id = StateId(retained);
            } else {
                error(
                    declaration,
                    analysis.span,
                    "`retain <name>` is ambiguous for a procedure that publishes multiple states",
                    diagnostics,
                );
            }
        }
        add_state_flow(
            declaration,
            &analysis,
            &step,
            &mut dependencies,
            &mut transitions,
            diagnostics,
        );
        if let Some(predecessor) = &analysis.predecessor {
            if !step.state_inputs.iter().any(|input| {
                matches!(&input.source, StateSource::Upstream { producer_step, .. } if producer_step.as_str() == predecessor)
            }) {
                dependencies.push(SequenceDependency {
                    predecessor: ProcedureStepId(predecessor.clone()),
                    successor: step.step_id.clone(),
                    reason: SequenceDependencyReason::AuthoredOrdering,
                });
            }
        }
        for input in &mut step.state_inputs {
            let StateSource::Upstream {
                producer_step,
                state_id,
            } = &input.source
            else {
                continue;
            };
            if input.use_as != StateUse::HeatSource {
                continue;
            }
            match steps
                .iter()
                .find(|earlier: &&ProcedureStep| &earlier.step_id == producer_step)
                .and_then(|producer| {
                    producer
                        .state_outputs
                        .iter()
                        .find(|output| &output.state_id == state_id)
                }) {
                Some(output)
                    if matches!(
                        output.kind,
                        StateKind::ElectromagneticStatic | StateKind::ElectromagneticComplex
                    ) =>
                {
                    input.required_kind = output.kind;
                }
                Some(_) => error(
                    declaration,
                    analysis.span,
                    format!(
                        "`heat_source` must name an electromagnetic state; `analysis.{}.state.{}` is not one",
                        producer_step.as_str(),
                        state_id.0
                    ),
                    diagnostics,
                ),
                None => error(
                    declaration,
                    analysis.span,
                    format!(
                        "`heat_source` names `analysis.{}.state.{}`, which no earlier analysis publishes (retain it with `retain <name>`)",
                        producer_step.as_str(),
                        state_id.0
                    ),
                    diagnostics,
                ),
            }
        }
        steps.push(step);
    }

    if diagnostics.len() != initial_errors {
        return None;
    }
    let sequence = AnalysisSequence {
        sequence_id: format!("{}.sequence", declaration.key.name),
        steps,
        dependencies,
        transitions,
        branches: Vec::new(),
    };
    if let Err(validation) = validate_analysis_sequence(&sequence) {
        error(
            declaration,
            declaration.span,
            validation.to_string(),
            diagnostics,
        );
        return None;
    }
    Some(sequence)
}

/// A procedure member that no lowering rule reads would otherwise be silently ignored
/// (for example a misspelled `time_step`); report it with its own source span.
fn reject_unknown_members(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    read: &std::collections::BTreeSet<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    for statement in statements {
        let CheckedFeaStatementKind::Assignment { target, .. } = &statement.kind else {
            continue;
        };
        let Some(name) = target.last() else {
            continue;
        };
        if !read.contains(name) {
            error(
                declaration,
                statement.span,
                crate::semantic::pending::with_pending(
                    format!(
                        "`{name}` is not a member of `{} {}`",
                        header.family, header.procedure
                    ),
                    crate::semantic::pending::procedure_member(name),
                ),
                diagnostics,
            );
        }
    }
}

fn procedure_block<'a>(
    declaration: &CheckedFeaDeclaration,
    statements: &'a [CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(ProcedureHeader, &'a [CheckedFeaStatement])> {
    let mut procedures = statements.iter().filter_map(|statement| {
        let CheckedFeaStatementKind::Block {
            header,
            header_span,
            statements,
        } = &statement.kind
        else {
            return None;
        };
        header::procedure(header, *header_span).map(|header| (header, statements.as_slice()))
    });
    let result = procedures.next();
    if result.is_none() {
        error(
            declaration,
            declaration.span,
            "analysis requires exactly one typed physics procedure block",
            diagnostics,
        );
    } else if procedures.next().is_some() {
        error(
            declaration,
            declaration.span,
            "analysis cannot contain more than one physics procedure block",
            diagnostics,
        );
        return None;
    }
    result
}

fn lower_procedure(
    declaration: &CheckedFeaDeclaration,
    analysis: &AnalysisHeader,
    procedure: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    models: &[&AuthoredModel],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ProcedureStep> {
    let id = analysis.name.clone();
    let step = match (procedure.family.as_str(), procedure.procedure.as_str()) {
        ("structural", "static") => ProcedureStep::structural_static(id),
        ("structural", "preload") => ProcedureStep::structural_preload(id),
        ("thermomechanical", "preload") => {
            let state = super::thermal_preload::temperature(
                declaration,
                procedure,
                statements,
                diagnostics,
            )?;
            ProcedureStep::thermomechanical_preload(id, state.0, state.1)
        }
        ("structural", "modal") => {
            let mut step = ProcedureStep::structural_modal(id);
            step.independent_domain = IndependentDomain::Eigenvalue {
                requested_count: requirements::count(
                    declaration,
                    procedure,
                    statements,
                    diagnostics,
                )?,
            };
            super::modal_reduction::extraction(
                declaration,
                statements,
                &mut step,
                diagnostics,
            )?;
            step
        }
        ("structural", "transient") => {
            let mut step = ProcedureStep::structural_transient(id);
            step.independent_domain =
                requirements::time(declaration, procedure, statements, diagnostics)?;
            step
        }
        ("structural", "modal_transient") => {
            let state =
                requirements::state(declaration, procedure, statements, "basis", diagnostics)?;
            let mut step = ProcedureStep::structural_modal_transient(id, state.0, state.1);
            step.independent_domain =
                requirements::time(declaration, procedure, statements, diagnostics)?;
            step
        }
        ("structural", "harmonic") => {
            let mut step = ProcedureStep::structural_harmonic(id);
            step.independent_domain =
                requirements::frequencies(declaration, procedure, statements, diagnostics)?;
            step
        }
        ("structural", "modal_harmonic") => {
            let state =
                requirements::state(declaration, procedure, statements, "basis", diagnostics)?;
            let mut step = ProcedureStep::structural_modal_harmonic(id, state.0, state.1);
            step.independent_domain =
                requirements::frequencies(declaration, procedure, statements, diagnostics)?;
            super::modal_reduction::static_correction(
                declaration,
                statements,
                &mut step,
                diagnostics,
            )?;
            step
        }
        ("structural", "prestressed_modal") => {
            let state =
                requirements::state(declaration, procedure, statements, "prestress", diagnostics)?;
            let mut step = ProcedureStep::structural_prestressed_modal(id, state.0, state.1);
            step.independent_domain = IndependentDomain::Eigenvalue {
                requested_count: requirements::count(
                    declaration,
                    procedure,
                    statements,
                    diagnostics,
                )?,
            };
            super::modal_reduction::extraction(
                declaration,
                statements,
                &mut step,
                diagnostics,
            )?;
            step
        }
        ("structural", "buckling") => {
            let state =
                requirements::state(declaration, procedure, statements, "prestress", diagnostics)?;
            let mut step = ProcedureStep::structural_buckling(id, state.0, state.1);
            step.independent_domain = IndependentDomain::Eigenvalue {
                requested_count: requirements::count(
                    declaration,
                    procedure,
                    statements,
                    diagnostics,
                )?,
            };
            step
        }
        ("structural", "prestressed_harmonic") => {
            super::prestressed_harmonic::lower(id, declaration, procedure, statements, diagnostics)?
        }
        ("structural", "modal_response_spectrum") => {
            let state =
                requirements::state(declaration, procedure, statements, "basis", diagnostics)?;
            let spectra: Vec<ResponseSpectrumId> = requirements::references(
                declaration,
                procedure,
                statements,
                "spectra",
                "response_spectrum",
                diagnostics,
            )?
            .into_iter()
            .map(ResponseSpectrumId)
            .collect();
            if spectra.iter().any(|id| {
                !models
                    .iter()
                    .flat_map(|model| &model.response_spectra)
                    .any(|spectrum| spectrum.spectrum_id == *id)
            }) {
                error(
                    declaration,
                    procedure.span,
                    "modal response references a response spectrum not owned by the bound model",
                    diagnostics,
                );
                return None;
            }
            ProcedureStep::structural_modal_response_spectrum(id, spectra, state.0, state.1)
        }
        ("structural", "random_response") => {
            let state =
                requirements::state(declaration, procedure, statements, "basis", diagnostics)?;
            let excitation = requirements::reference(
                declaration,
                procedure,
                statements,
                "excitation",
                "random_excitation",
                diagnostics,
            )?;
            if !models
                .iter()
                .flat_map(|model| &model.random_excitations)
                .any(|value| value.excitation_id.0 == excitation)
            {
                error(
                    declaration,
                    procedure.span,
                    "random response references a random excitation not owned by the bound model",
                    diagnostics,
                );
                return None;
            }
            let settings = StructuralRandomResponseSettings {
                modal_damping_ratio: requirements::ratio(
                    declaration,
                    procedure,
                    statements,
                    "modal_damping_ratio",
                    false,
                    false,
                    diagnostics,
                )?,
                minimum_cumulative_mass_ratio: requirements::ratio(
                    declaration,
                    procedure,
                    statements,
                    "minimum_cumulative_mass_ratio",
                    true,
                    true,
                    diagnostics,
                )?,
            };
            ProcedureStep::structural_random_response(
                id,
                RandomExcitationId(excitation),
                settings,
                state.0,
                state.1,
            )
        }
        ("structural", "nonlinear") => {
            let kinematics = match values::assignment(statements, "kinematics").map(|v| &v.kind) {
                None => None,
                Some(CheckedExpressionKind::Enum(name)) if name == "small_strain" => {
                    Some(StructuralKinematics::SmallStrain)
                }
                Some(CheckedExpressionKind::Enum(name)) if name == "large_deformation" => {
                    Some(StructuralKinematics::LargeDeformation)
                }
                Some(_) => {
                    error(
                        declaration,
                        procedure.span,
                        "structural nonlinear `kinematics` must be `small_strain` or `large_deformation`",
                        diagnostics,
                    );
                    return None;
                }
            };
            let mut step = match kinematics {
                Some(kinematics) => ProcedureStep::structural_nonlinear_with(
                    id,
                    StructuralNonlinearSettings { kinematics },
                ),
                None => ProcedureStep::structural_nonlinear(id),
            };
            step.independent_domain =
                requirements::load_factor(declaration, procedure, statements, diagnostics)?;
            step
        }
        ("thermal", "steady") => {
            let mut step = ProcedureStep::thermal_steady(id);
            heat_source(declaration, procedure, statements, &mut step, diagnostics)?;
            step
        }
        ("thermal", "transient") => {
            let mut step = ProcedureStep::thermal_transient(id);
            step.independent_domain =
                requirements::time(declaration, procedure, statements, diagnostics)?;
            heat_source(declaration, procedure, statements, &mut step, diagnostics)?;
            step
        }
        (family, name) => generic_procedure(
            id,
            family,
            name,
            declaration,
            procedure,
            statements,
            diagnostics,
        )?,
    };
    super::initial_conditions::bind(
        declaration,
        procedure,
        statements,
        models,
        step,
        diagnostics,
    )
}

/// `heat_source = analysis.<producer>.state.<name>`: the Joule loss density of an upstream
/// electromagnetic state heats this thermal step. The state kind is bound to the producer's
/// published state after every analysis is lowered.
fn heat_source(
    declaration: &CheckedFeaDeclaration,
    procedure: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    step: &mut ProcedureStep,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<()> {
    if values::assignment(statements, "heat_source").is_none() {
        return Some(());
    }
    let (producer, state) = requirements::state(
        declaration,
        procedure,
        statements,
        "heat_source",
        diagnostics,
    )?;
    step.state_inputs.push(StateInput {
        input_id: "heat_source".into(),
        required_kind: StateKind::ElectromagneticComplex,
        source: StateSource::Upstream {
            producer_step: ProcedureStepId(producer),
            state_id: StateId(state),
        },
        use_as: StateUse::HeatSource,
        conflict: InitialStateConflictPolicy::Reject,
    });
    Some(())
}

fn generic_procedure(
    id: String,
    family: &str,
    name: &str,
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ProcedureStep> {
    let family = match family {
        "electromagnetic" => PhysicsFamily::Electromagnetic,
        "acoustic" => PhysicsFamily::Acoustic,
        "fluid" => PhysicsFamily::Fluid,
        "thermomechanical" => PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
        "electrothermal" => PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal),
        "conjugate_heat_transfer" => {
            PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer)
        }
        "fluid_structure_interaction" => {
            PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction)
        }
        _ => return unsupported(declaration, header, diagnostics),
    };
    let (procedure, domain, output) = match name {
        "static" | "steady" => (
            AnalysisProcedure::Static,
            IndependentDomain::Static,
            static_state(family),
        ),
        "transient" => (
            AnalysisProcedure::TransientResponse,
            requirements::time(declaration, header, statements, diagnostics)?,
            transient_state(family),
        ),
        "harmonic" => (
            AnalysisProcedure::HarmonicResponse,
            requirements::frequencies(declaration, header, statements, diagnostics)?,
            harmonic_state(family),
        ),
        "modal" if family == PhysicsFamily::Acoustic => (
            AnalysisProcedure::ModalExtraction,
            IndependentDomain::Eigenvalue {
                requested_count: requirements::count(declaration, header, statements, diagnostics)?,
            },
            StateKind::AcousticModes,
        ),
        // The ledger identity row owns its status; admission reports it while it is pending.
        "line_parameters" if family == PhysicsFamily::Electromagnetic => (
            AnalysisProcedure::LineParameters,
            IndependentDomain::Static,
            static_state(family),
        ),
        "nonlinear" => (
            AnalysisProcedure::Nonlinear,
            requirements::load_factor(declaration, header, statements, diagnostics)?,
            static_state(family),
        ),
        _ => return unsupported(declaration, header, diagnostics),
    };
    Some(ProcedureStep::new(
        id,
        family,
        procedure,
        SolutionSpace::Direct,
        domain,
        vec![state_output("primary", output)],
    ))
}

fn add_state_flow(
    declaration: &CheckedFeaDeclaration,
    analysis: &AnalysisHeader,
    step: &ProcedureStep,
    dependencies: &mut Vec<SequenceDependency>,
    transitions: &mut Vec<StateTransition>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let producers = step
        .state_inputs
        .iter()
        .filter_map(|input| match &input.source {
            StateSource::Upstream { producer_step, .. } => Some(producer_step.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !producers.is_empty()
        && !analysis
            .predecessor
            .as_deref()
            .is_some_and(|after| producers.contains(&after))
    {
        error(
            declaration,
            analysis.span,
            format!("analysis `{}` must declare `after analysis.<producer>` naming one of its consumed state producers", analysis.name),
            diagnostics,
        );
    }
    for input in &step.state_inputs {
        let StateSource::Upstream {
            producer_step,
            state_id,
        } = &input.source
        else {
            continue;
        };
        dependencies.push(SequenceDependency {
            predecessor: producer_step.clone(),
            successor: step.step_id.clone(),
            reason: SequenceDependencyReason::StateTransfer,
        });
        // Inputs name already published states: temperature, geometric stiffness
        // and modal basis are reused, not re-derived by transferring them. The
        // producer owns any linearization; consumers preserve its reference state.
        let (method, reference) = match input.use_as {
            StateUse::InitialGuess => (
                StateTransitionMethod::InitialGuessOnly,
                ReferenceStatePolicy::Preserve,
            ),
            _ => (StateTransitionMethod::Reuse, ReferenceStatePolicy::Preserve),
        };
        transitions.push(StateTransition {
            transition_id: TransitionId(format!(
                "{}_to_{}_{}",
                producer_step.as_str(),
                step.step_id.as_str(),
                state_id.0
            )),
            producer_step: producer_step.clone(),
            consumer_step: step.step_id.clone(),
            state_id: state_id.clone(),
            method,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference,
        });
    }
}

fn state_output(id: &str, kind: StateKind) -> StateOutput {
    StateOutput {
        state_id: StateId(id.to_string()),
        kind,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::UntilDependentsComplete,
    }
}

fn static_state(family: PhysicsFamily) -> StateKind {
    match family {
        PhysicsFamily::Structural => StateKind::StructuralPrimary,
        PhysicsFamily::Thermal => StateKind::Thermal,
        PhysicsFamily::Electromagnetic => StateKind::ElectromagneticStatic,
        PhysicsFamily::Acoustic => StateKind::AcousticComplex,
        PhysicsFamily::Fluid => StateKind::Fluid,
        // A thermomechanical step solves the structure under its temperature field, and an
        // electrothermal step the temperature under Joule heating; their primary states are
        // those solutions. Other couplings exchange interface state.
        PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical) => {
            StateKind::StructuralPrimary
        }
        PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::Electrothermal) => {
            StateKind::Thermal
        }
        PhysicsFamily::Coupled(_) => StateKind::CoupledInterface,
    }
}

fn transient_state(family: PhysicsFamily) -> StateKind {
    match family {
        PhysicsFamily::Acoustic => StateKind::AcousticPressure,
        _ => static_state(family),
    }
}

fn harmonic_state(family: PhysicsFamily) -> StateKind {
    match family {
        PhysicsFamily::Structural => StateKind::StructuralComplex,
        PhysicsFamily::Electromagnetic => StateKind::ElectromagneticComplex,
        PhysicsFamily::Acoustic => StateKind::AcousticComplex,
        _ => static_state(family),
    }
}

fn unsupported(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ProcedureStep> {
    error(
        declaration,
        header.span,
        format!(
            "unsupported procedure `{} {}`",
            header.family, header.procedure
        ),
        diagnostics,
    );
    None
}

pub(super) fn error(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.procedure",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    ));
}
