use super::{header::ProcedureHeader, lower::error, values};
use crate::{CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement, FeaDiagnostic};
use runmat_analysis_core::{
    physics_family_can_consume_state, ActivationLifecycle, AuthoredInitialConditionKind,
    AuthoredModel, InitialStateConflictPolicy, ProcedureStep, StateInput, StateKind, StateSource,
    StateUse, StepObjectActivation, StepObjectRef,
};
use std::collections::HashSet;

pub(super) fn bind(
    declaration: &CheckedFeaDeclaration,
    procedure: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    models: &[&AuthoredModel],
    mut step: ProcedureStep,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ProcedureStep> {
    let Some(expression) = values::assignment(statements, "initial_conditions") else {
        return Some(step);
    };
    let CheckedExpressionKind::List(references) = &expression.kind else {
        error(
            declaration,
            expression.span,
            "`initial_conditions` requires a non-empty list of `model.initial.<name>` references",
            diagnostics,
        );
        return None;
    };
    if references.is_empty() {
        error(
            declaration,
            expression.span,
            "`initial_conditions` cannot be empty",
            diagnostics,
        );
        return None;
    }
    let mut ids = HashSet::new();
    let mut state_kinds = HashSet::new();
    for reference in references {
        let CheckedExpressionKind::Reference(path) = &reference.kind else {
            return invalid_reference(declaration, reference.span, diagnostics);
        };
        let [model_namespace, initial_namespace, id] = path.as_slice() else {
            return invalid_reference(declaration, reference.span, diagnostics);
        };
        if model_namespace != "model" || initial_namespace != "initial" || !ids.insert(id) {
            return invalid_reference(declaration, reference.span, diagnostics);
        }
        let Some(condition) = models
            .iter()
            .flat_map(|model| &model.initial_conditions)
            .find(|condition| condition.initial_condition_id == *id)
        else {
            error(
                declaration,
                reference.span,
                format!("model has no initial condition `{id}`"),
                diagnostics,
            );
            return None;
        };
        let (state_kind, input_id) = state_binding(&condition.kind);
        if !physics_family_can_consume_state(step.family, state_kind) {
            error(
                declaration,
                reference.span,
                format!(
                    "procedure `{} {}` cannot consume initial state {:?}",
                    procedure.family, procedure.procedure, state_kind
                ),
                diagnostics,
            );
            return None;
        }
        step.activations.push(StepObjectActivation {
            object: StepObjectRef::InitialCondition(id.clone()),
            lifecycle: ActivationLifecycle::Activate,
        });
        if state_kinds.insert(state_kind) {
            step.state_inputs.push(StateInput {
                input_id: input_id.to_owned(),
                required_kind: state_kind,
                source: StateSource::InitialConditions,
                use_as: StateUse::PhysicalInitialState,
                conflict: InitialStateConflictPolicy::Reject,
            });
        }
    }
    Some(step)
}

fn state_binding(kind: &AuthoredInitialConditionKind) -> (StateKind, &'static str) {
    match kind {
        AuthoredInitialConditionKind::Displacement(_) => {
            (StateKind::StructuralPrimary, "initial_structural_primary")
        }
        AuthoredInitialConditionKind::Velocity(_)
        | AuthoredInitialConditionKind::AngularVelocity(_) => (
            StateKind::StructuralKinematics,
            "initial_structural_kinematics",
        ),
        AuthoredInitialConditionKind::Temperature(_) => (StateKind::Thermal, "initial_thermal"),
        AuthoredInitialConditionKind::ElectricPotential(_)
        | AuthoredInitialConditionKind::MagneticVectorPotential(_) => (
            StateKind::ElectromagneticStatic,
            "initial_electromagnetic_static",
        ),
        AuthoredInitialConditionKind::AcousticPressure(_) => {
            (StateKind::AcousticComplex, "initial_acoustic_complex")
        }
        AuthoredInitialConditionKind::FluidPressure(_)
        | AuthoredInitialConditionKind::FluidVelocity(_) => (StateKind::Fluid, "initial_fluid"),
    }
}

fn invalid_reference(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ProcedureStep> {
    error(
        declaration,
        span,
        "initial conditions must be unique `model.initial.<name>` references",
        diagnostics,
    );
    None
}
