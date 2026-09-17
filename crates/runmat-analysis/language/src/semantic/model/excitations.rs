use super::{diagnostic, region_source, signals};
use crate::{
    BoundFeaProgram, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatementKind,
    FeaDiagnostic,
};
use runmat_analysis_core::{
    AuthoredExcitation, AuthoredExcitationCase, AuthoredExcitationKind, AuthoredExcitationTarget,
    CoordinateFrame, ExcitationCaseActivation, ExcitationKinematics, PhysicalSignal,
    ProjectionRequirement, SignalId,
};
use runmat_parser::fea::FeaModelDefinitionKind;
use std::collections::BTreeSet;

pub(super) struct LoweredExcitations {
    pub signals: Vec<PhysicalSignal>,
    pub excitations: Vec<AuthoredExcitation>,
    pub cases: Vec<AuthoredExcitationCase>,
}

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> LoweredExcitations {
    let mut signals = Vec::new();
    let mut excitations = Vec::new();
    let mut cases = Vec::new();
    for statement in &declaration.statements {
        let CheckedFeaStatementKind::ModelDefinition {
            kind,
            name,
            statements,
        } = &statement.kind
        else {
            continue;
        };
        match kind {
            FeaModelDefinitionKind::Load => {
                if is_applied_load(statements) {
                    if let Some((signal, excitation)) =
                        load(declaration, bound, name, statements, diagnostics)
                    {
                        signals.push(signal);
                        excitations.push(excitation);
                    }
                }
            }
            FeaModelDefinitionKind::LoadCase => {
                if let Some(case) = load_case(declaration, name, statements, diagnostics) {
                    cases.push(case);
                }
            }
            _ => {}
        }
    }
    LoweredExcitations {
        signals,
        excitations,
        cases,
    }
}

fn is_applied_load(statements: &[crate::CheckedFeaStatement]) -> bool {
    matches!(
        statements,
        [crate::CheckedFeaStatement {
            kind: CheckedFeaStatementKind::AppliedLoad { .. },
            ..
        }]
    )
}

fn load(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    name: &str,
    statements: &[crate::CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(PhysicalSignal, AuthoredExcitation)> {
    let [statement] = statements else {
        diagnostic(
            declaration,
            declaration.span,
            "load requires exactly one applied action",
            diagnostics,
        );
        return None;
    };
    let CheckedFeaStatementKind::AppliedLoad {
        kind,
        target,
        value,
    } = &statement.kind
    else {
        diagnostic(
            declaration,
            statement.span,
            "load requires `<kind> on <target> = <value>`",
            diagnostics,
        );
        return None;
    };
    let (kind, quantity) = kind_policy(declaration, kind, statement.span, diagnostics)?;
    let signal_id = format!("load.{name}.value");
    let signal =
        signals::constant_from_expression(declaration, &signal_id, quantity, value, diagnostics)?;
    let target =
        AuthoredExcitationTarget::Region(region_source(bound, declaration, target, diagnostics)?);
    let frame = signal.frame().cloned().unwrap_or(CoordinateFrame::Global);
    Some((
        signal,
        AuthoredExcitation {
            excitation_id: format!("load.{name}"),
            kind,
            target,
            frame,
            direction: None,
            distribution: kind.default_distribution(),
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId(signal_id),
            active_step_ids: Vec::new(),
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: kind.preserves_resultant(),
                preserve_moment: kind.preserves_moment(),
                relative_tolerance: 1.0e-10,
            },
        },
    ))
}

fn load_case(
    declaration: &CheckedFeaDeclaration,
    name: &str,
    statements: &[crate::CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredExcitationCase> {
    let mut ids = Vec::new();
    let mut seen = BTreeSet::new();
    for statement in statements {
        let CheckedFeaStatementKind::Include(value) = &statement.kind else {
            diagnostic(
                declaration,
                statement.span,
                "load case accepts only `include load.<name>` members",
                diagnostics,
            );
            continue;
        };
        let CheckedExpressionKind::Reference(path) = &value.kind else {
            diagnostic(
                declaration,
                value.span,
                "load case include requires a load reference",
                diagnostics,
            );
            continue;
        };
        let [namespace, id] = path.as_slice() else {
            diagnostic(
                declaration,
                value.span,
                "load case include requires `load.<name>`",
                diagnostics,
            );
            continue;
        };
        if namespace != "load" || !seen.insert(id.as_str()) {
            diagnostic(
                declaration,
                value.span,
                "load case members must be unique load references",
                diagnostics,
            );
            continue;
        }
        ids.push(format!("load.{id}"));
    }
    Some(AuthoredExcitationCase {
        case_id: format!("load_case.{name}"),
        excitation_ids: ids,
        activation: ExcitationCaseActivation::Always,
    })
}

fn kind_policy(
    declaration: &CheckedFeaDeclaration,
    kind: &str,
    span: runmat_types::Span,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(AuthoredExcitationKind, &'static str)> {
    Some(match kind {
        "force" => (AuthoredExcitationKind::Force, "force"),
        "moment" => (AuthoredExcitationKind::Moment, "moment"),
        "traction" => (AuthoredExcitationKind::Traction, "traction"),
        "pressure" => (AuthoredExcitationKind::Pressure, "pressure"),
        "body_force" => (AuthoredExcitationKind::BodyForce, "body_force_density"),
        "heat_generation" => (
            AuthoredExcitationKind::HeatGeneration,
            "heat_generation_density",
        ),
        "heat_flux" => (AuthoredExcitationKind::HeatFlux, "heat_flux"),
        "temperature_field" => (AuthoredExcitationKind::TemperatureField, "temperature"),
        "ambient_temperature" => (AuthoredExcitationKind::AmbientTemperature, "temperature"),
        "terminal_voltage" => (
            AuthoredExcitationKind::TerminalVoltage,
            "electric_potential",
        ),
        "electric_current" => (AuthoredExcitationKind::ElectricCurrent, "electric_current"),
        "electric_charge" => (AuthoredExcitationKind::ElectricCharge, "electric_charge"),
        "current_density" => (
            AuthoredExcitationKind::CurrentDensity,
            "electric_current_density",
        ),
        "magnetic_vector_potential" => (
            AuthoredExcitationKind::MagneticVectorPotential,
            "magnetic_vector_potential",
        ),
        "acoustic_pressure" => (AuthoredExcitationKind::AcousticPressure, "pressure"),
        "normal_velocity" => (AuthoredExcitationKind::NormalVelocity, "velocity"),
        "inlet_velocity" => (AuthoredExcitationKind::InletVelocity, "velocity"),
        "mass_flow" => (AuthoredExcitationKind::MassFlow, "mass_flow_rate"),
        "wall_motion" => (AuthoredExcitationKind::WallMotion, "velocity"),
        _ => {
            diagnostic(
                declaration,
                span,
                format!("unsupported applied load kind `{kind}`"),
                diagnostics,
            );
            return None;
        }
    })
}
