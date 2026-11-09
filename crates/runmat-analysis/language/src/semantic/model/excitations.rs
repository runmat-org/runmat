use super::{diagnostic, properties::*, region_source, signals};
use crate::{
    BoundFeaProgram, CheckedCallArgument, CheckedExpression, CheckedExpressionKind,
    CheckedFeaDeclaration, CheckedFeaStatementKind, FeaDiagnostic,
};
use runmat_analysis_core::{
    AuthoredExcitation, AuthoredExcitationCase, AuthoredExcitationKind, AuthoredExcitationTarget,
    AuthoredParameterizedSignal, CoordinateFrame, ExcitationCaseActivation, ExcitationDistribution,
    ExcitationKinematics, PhysicalSignal, ProjectionRequirement, SignalId,
};
use runmat_fea_syntax::FeaModelDefinitionKind;
use std::collections::BTreeSet;

pub(super) struct LoweredExcitations {
    pub signals: Vec<PhysicalSignal>,
    pub parameterized_signals: Vec<AuthoredParameterizedSignal>,
    pub excitations: Vec<AuthoredExcitation>,
    pub cases: Vec<AuthoredExcitationCase>,
}

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    declared_signals: &[PhysicalSignal],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> LoweredExcitations {
    let mut signals = Vec::new();
    let mut parameterized_signals = Vec::new();
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
                        match signal {
                            LoweredLoadSignal::Concrete(signal) => signals.push(*signal),
                            LoweredLoadSignal::Parameterized(signal) => {
                                parameterized_signals.push(*signal)
                            }
                        }
                        excitations.push(excitation);
                    }
                }
            }
            FeaModelDefinitionKind::LoadCase => {
                if let Some(case) = load_case(declaration, name, statements, diagnostics) {
                    cases.push(case);
                }
            }
            FeaModelDefinitionKind::Excitation => {
                if let Some(excitation) = explicit_excitation(
                    declaration,
                    bound,
                    declared_signals,
                    name,
                    statements,
                    diagnostics,
                ) {
                    excitations.push(excitation);
                }
            }
            _ => {}
        }
    }
    if cases.is_empty() && !excitations.is_empty() && !declares_load_case(declaration) {
        cases.push(default_case(&excitations));
    }
    LoweredExcitations {
        signals,
        parameterized_signals,
        excitations,
        cases,
    }
}

/// Case id of the implicit case that holds every load of a model that declares no `load_case`.
pub const DEFAULT_LOAD_CASE_ID: &str = "load_case.default";

fn declares_load_case(declaration: &CheckedFeaDeclaration) -> bool {
    declaration.statements.iter().any(|statement| {
        matches!(
            &statement.kind,
            CheckedFeaStatementKind::ModelDefinition {
                kind: FeaModelDefinitionKind::LoadCase,
                ..
            }
        )
    })
}

/// A model without any `load_case` applies all of its loads together in one default case, so a
/// minimal model needs no case declaration. Once a model declares a case, every load must be
/// placed in a declared case explicitly.
fn default_case(excitations: &[AuthoredExcitation]) -> AuthoredExcitationCase {
    AuthoredExcitationCase {
        case_id: DEFAULT_LOAD_CASE_ID.to_string(),
        excitation_ids: excitations
            .iter()
            .map(|excitation| excitation.excitation_id.clone())
            .collect(),
        activation: ExcitationCaseActivation::Always,
    }
}

fn explicit_excitation(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    signals: &[PhysicalSignal],
    name: &str,
    statements: &[crate::CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredExcitation> {
    let (kind_name, body) = sole_kind_block(declaration, statements, "excitation", diagnostics)?;
    let (kind, _) = kind_policy(declaration, kind_name, declaration.span, diagnostics)?;
    validate_assignments(
        declaration,
        body,
        &["target", "signal", "direction", "kinematics"],
        diagnostics,
    );
    let signal_expression = required_assignment(declaration, body, "signal", diagnostics)?;
    let signal_id = signal_reference(declaration, signal_expression, diagnostics)?;
    let signal = signals
        .iter()
        .find(|signal| signal.signal_id() == &signal_id)
        .or_else(|| {
            diagnostic(
                declaration,
                signal_expression.span,
                format!("excitation references unknown signal `{}`", signal_id.0),
                diagnostics,
            );
            None
        })?;
    let target_expression = required_assignment(declaration, body, "target", diagnostics)?;
    let target = if matches!(&target_expression.kind, CheckedExpressionKind::Enum(name) if name == "whole_model")
    {
        AuthoredExcitationTarget::WholeModel
    } else if let Some(constraint_id) = coupling_reference_point(target_expression) {
        AuthoredExcitationTarget::ReferencePoint { constraint_id }
    } else {
        AuthoredExcitationTarget::Region(region_source(
            bound,
            declaration,
            target_expression,
            diagnostics,
        )?)
    };
    let direction = match assignment(body, "direction") {
        Some(value) => Some(direction(declaration, value, diagnostics)?),
        None => None,
    };
    let frame = match (&direction, signal.frame()) {
        (Some((_, frame)), _) => frame.clone(),
        (None, Some(frame)) => frame.clone(),
        (None, None) => CoordinateFrame::Global,
    };
    let direction = direction.map(|(direction, _)| direction);
    let kinematics = match assignment(body, "kinematics").and_then(enum_name) {
        None | Some("fixed_frame") => ExcitationKinematics::FixedFrame,
        Some("follower") => ExcitationKinematics::Follower,
        Some(_) => {
            diagnostic(
                declaration,
                declaration.span,
                "excitation kinematics must be `fixed_frame` or `follower`",
                diagnostics,
            );
            return None;
        }
    };
    Some(AuthoredExcitation {
        excitation_id: format!("excitation.{name}"),
        kind,
        target,
        frame,
        direction,
        distribution: if signal.shape().topology_component_count().is_some() {
            ExcitationDistribution::TopologyField
        } else if kind == AuthoredExcitationKind::HeatGeneration
            && signal.quantity() == runmat_analysis_core::PhysicalQuantityKind::Power
        {
            ExcitationDistribution::PreserveResultant
        } else {
            kind.default_distribution()
        },
        kinematics,
        signal_id,
        active_step_ids: Vec::new(),
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: kind.preserves_resultant(),
            preserve_moment: kind.preserves_moment(),
            relative_tolerance: 1.0e-10,
        },
    })
}

fn signal_reference(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SignalId> {
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        diagnostic(
            declaration,
            expression.span,
            "excitation signal must reference `signal.<name>`",
            diagnostics,
        );
        return None;
    };
    let [namespace, name] = path.as_slice() else {
        diagnostic(
            declaration,
            expression.span,
            "excitation signal must reference `signal.<name>`",
            diagnostics,
        );
        return None;
    };
    if namespace != "signal" {
        diagnostic(
            declaration,
            expression.span,
            "excitation signal must reference `signal.<name>`",
            diagnostics,
        );
        return None;
    }
    Some(SignalId(name.clone()))
}

fn direction(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(runmat_analysis_core::Direction, CoordinateFrame)> {
    let CheckedExpressionKind::OperationCall {
        lowering,
        arguments,
        ..
    } = &expression.kind
    else {
        diagnostic(
            declaration,
            expression.span,
            "excitation direction requires `global([x, y, z])`",
            diagnostics,
        );
        return None;
    };
    if lowering.as_str() != "fea.geometry.global" || arguments.len() != 1 {
        diagnostic(
            declaration,
            expression.span,
            "excitation direction requires `global([x, y, z])`",
            diagnostics,
        );
        return None;
    }
    let value = match &arguments[0] {
        CheckedCallArgument::Positional(value) => value,
        CheckedCallArgument::Named { value, .. } => value,
    };
    let CheckedExpressionKind::List(values) = &value.kind else {
        diagnostic(
            declaration,
            value.span,
            "excitation direction requires three constant components",
            diagnostics,
        );
        return None;
    };
    if values.len() != 3 {
        diagnostic(
            declaration,
            value.span,
            "excitation direction requires three constant components",
            diagnostics,
        );
        return None;
    }
    let vector = [
        scalar(&values[0])?,
        scalar(&values[1])?,
        scalar(&values[2])?,
    ];
    Some((
        runmat_analysis_core::Direction { vector },
        CoordinateFrame::Global,
    ))
}

fn assignment<'a>(
    body: &'a [crate::CheckedFeaStatement],
    name: &str,
) -> Option<&'a CheckedExpression> {
    body.iter().find_map(|statement| match &statement.kind {
        CheckedFeaStatementKind::Assignment { target, value, .. }
            if target.as_slice() == [name] =>
        {
            Some(value)
        }
        _ => None,
    })
}

fn enum_name(expression: &CheckedExpression) -> Option<&str> {
    match &expression.kind {
        CheckedExpressionKind::Enum(name) => Some(name),
        _ => None,
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
) -> Option<(LoweredLoadSignal, AuthoredExcitation)> {
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
    // `heat_generation on <solid> = <W>` is a total heat rate, distributed uniformly over the
    // region's meshed volume when the study compiles; W/m^3 stays a density.
    let total_heat_rate = kind == AuthoredExcitationKind::HeatGeneration
        && crate::semantic::units::is_dimension(value, runmat_types::PhysicalDimension::POWER);
    let (quantity, distribution) = if total_heat_rate {
        ("heat_rate", ExcitationDistribution::PreserveResultant)
    } else {
        (quantity, kind.default_distribution())
    };
    let signal_id = format!("load.{name}.value");
    let signal = if signals::depends_on_model_input(value) {
        LoweredLoadSignal::Parameterized(Box::new(signals::parameterized_from_expression(
            declaration,
            &signal_id,
            kind.signal_quantity(distribution),
            quantity,
            value,
            diagnostics,
        )?))
    } else {
        LoweredLoadSignal::Concrete(Box::new(signals::constant_from_expression(
            declaration,
            &signal_id,
            quantity,
            value,
            diagnostics,
        )?))
    };
    let target = match coupling_reference_point(target) {
        Some(constraint_id) => AuthoredExcitationTarget::ReferencePoint { constraint_id },
        None => AuthoredExcitationTarget::Region(region_source(
            bound,
            declaration,
            target,
            diagnostics,
        )?),
    };
    let frame = match &signal {
        LoweredLoadSignal::Concrete(signal) => signal.frame().cloned(),
        LoweredLoadSignal::Parameterized(signal) => signal.frame.clone(),
    }
    .unwrap_or(CoordinateFrame::Global);
    Some((
        signal,
        AuthoredExcitation {
            excitation_id: format!("load.{name}"),
            kind,
            target,
            frame,
            direction: None,
            distribution,
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

/// `constraint.<coupling>.reference_point`: the reference point a rigid or distributing coupling
/// defines. Model realization binds it to that coupling's identity and position.
fn coupling_reference_point(expression: &CheckedExpression) -> Option<String> {
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        return None;
    };
    match path.as_slice() {
        [namespace, constraint, member]
            if namespace == "constraint" && member == "reference_point" =>
        {
            Some(constraint.clone())
        }
        _ => None,
    }
}

enum LoweredLoadSignal {
    Concrete(Box<PhysicalSignal>),
    Parameterized(Box<AuthoredParameterizedSignal>),
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
                "load case accepts only `include load.<name>` or `include excitation.<name>` members",
                diagnostics,
            );
            continue;
        };
        let CheckedExpressionKind::Reference(path) = &value.kind else {
            diagnostic(
                declaration,
                value.span,
                "load case include requires a load or excitation reference",
                diagnostics,
            );
            continue;
        };
        let [namespace, id] = path.as_slice() else {
            diagnostic(
                declaration,
                value.span,
                "load case include requires `load.<name>` or `excitation.<name>`",
                diagnostics,
            );
            continue;
        };
        if !matches!(namespace.as_str(), "load" | "excitation")
            || !seen.insert((namespace.as_str(), id.as_str()))
        {
            diagnostic(
                declaration,
                value.span,
                "load case members must be unique load or excitation references",
                diagnostics,
            );
            continue;
        }
        ids.push(format!("{namespace}.{id}"));
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
        "electric_charge_density" => (
            AuthoredExcitationKind::ElectricChargeDensity,
            "electric_charge_density",
        ),
        "current_density" => (
            AuthoredExcitationKind::CurrentDensity,
            "electric_current_density",
        ),
        "magnetic_volume_current_density" => (
            AuthoredExcitationKind::MagneticVolumeCurrentDensity,
            "electric_current_density",
        ),
        "magnetic_vector_potential" => (
            AuthoredExcitationKind::MagneticVectorPotential,
            "magnetic_vector_potential",
        ),
        "acoustic_pressure" => (AuthoredExcitationKind::AcousticPressure, "pressure"),
        "incident_pressure" => (AuthoredExcitationKind::AcousticIncidentPressure, "pressure"),
        "normal_velocity" => (AuthoredExcitationKind::NormalVelocity, "velocity"),
        "volume_velocity" => (AuthoredExcitationKind::VolumeVelocity, "volume_flow_rate"),
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
