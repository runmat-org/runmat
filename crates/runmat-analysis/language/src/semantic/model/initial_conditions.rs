use super::{diagnostic, properties::*, region_source};
use crate::{
    BoundFeaProgram, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaDiagnostic,
};
use runmat_analysis_core::{
    AngularVelocity, AuthoredInitialCondition, AuthoredInitialConditionKind,
    AuthoredInitialConditionTarget, CoordinateFrame, Displacement, ElectricPotential,
    MagneticVectorPotential, PhysicalQuantityKind, PhysicalSignal, Pressure, Temperature, Velocity,
};
use runmat_design_syntax::FeaAssignmentRelation;
use runmat_design_syntax::FeaModelDefinitionKind;
use std::collections::BTreeSet;

mod values;

use values::{parse_frame, scalar_value, vector_value, ValueContract};

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    signals: &[PhysicalSignal],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<AuthoredInitialCondition> {
    declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::ModelDefinition {
                kind: FeaModelDefinitionKind::Initial,
                name,
                statements,
            } = &statement.kind
            else {
                return None;
            };
            condition(declaration, bound, signals, name, statements, diagnostics)
        })
        .collect()
}

fn condition(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    signals: &[PhysicalSignal],
    id: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredInitialCondition> {
    let (kind, body) = sole_kind_block(declaration, statements, "initial condition", diagnostics)?;
    let vector = matches!(
        kind,
        "structural.displacement"
            | "structural.velocity"
            | "structural.angular_velocity"
            | "magnetic.vector_potential"
            | "fluid.velocity"
    );
    validate_members(declaration, body, vector, diagnostics);
    let target_expression = required_assignment(declaration, body, "target", diagnostics)?;
    let target = if matches!(&target_expression.kind, CheckedExpressionKind::Enum(name) if name == "whole_model")
    {
        AuthoredInitialConditionTarget::WholeModel
    } else {
        AuthoredInitialConditionTarget::Region(region_source(
            bound,
            declaration,
            target_expression,
            diagnostics,
        )?)
    };
    let frame = if vector {
        parse_frame(
            declaration,
            required_assignment(declaration, body, "frame", diagnostics)?,
            diagnostics,
        )?
    } else {
        CoordinateFrame::Global
    };
    let kind = match kind {
        "structural.displacement" => AuthoredInitialConditionKind::Displacement(vector_value(
            declaration,
            body,
            signals,
            ValueContract::new(PhysicalQuantityKind::Displacement),
            Displacement::new,
            diagnostics,
        )?),
        "structural.velocity" => AuthoredInitialConditionKind::Velocity(vector_value(
            declaration,
            body,
            signals,
            ValueContract::new(PhysicalQuantityKind::Velocity),
            Velocity::new,
            diagnostics,
        )?),
        "structural.angular_velocity" => {
            AuthoredInitialConditionKind::AngularVelocity(vector_value(
                declaration,
                body,
                signals,
                ValueContract::new(PhysicalQuantityKind::AngularVelocity),
                AngularVelocity::new,
                diagnostics,
            )?)
        }
        "thermal.temperature" => AuthoredInitialConditionKind::Temperature(scalar_value(
            declaration,
            body,
            signals,
            ValueContract::new(PhysicalQuantityKind::Temperature),
            Temperature::new,
            diagnostics,
        )?),
        "electrical.potential" => AuthoredInitialConditionKind::ElectricPotential(scalar_value(
            declaration,
            body,
            signals,
            ValueContract::new(PhysicalQuantityKind::ElectricPotential),
            ElectricPotential::new,
            diagnostics,
        )?),
        "magnetic.vector_potential" => {
            AuthoredInitialConditionKind::MagneticVectorPotential(vector_value(
                declaration,
                body,
                signals,
                ValueContract::new(PhysicalQuantityKind::MagneticVectorPotential),
                MagneticVectorPotential::new,
                diagnostics,
            )?)
        }
        "acoustic.pressure" => AuthoredInitialConditionKind::AcousticPressure(scalar_value(
            declaration,
            body,
            signals,
            ValueContract::new(PhysicalQuantityKind::Pressure),
            Pressure::new,
            diagnostics,
        )?),
        "fluid.pressure" => AuthoredInitialConditionKind::FluidPressure(scalar_value(
            declaration,
            body,
            signals,
            ValueContract::new(PhysicalQuantityKind::Pressure),
            Pressure::new,
            diagnostics,
        )?),
        "fluid.velocity" => AuthoredInitialConditionKind::FluidVelocity(vector_value(
            declaration,
            body,
            signals,
            ValueContract::new(PhysicalQuantityKind::Velocity),
            Velocity::new,
            diagnostics,
        )?),
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                format!("unsupported initial-condition kind `{kind}`"),
                diagnostics,
            );
            return None;
        }
    };
    Some(AuthoredInitialCondition {
        initial_condition_id: id.to_string(),
        target,
        frame,
        kind,
    })
}

fn validate_members(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    vector: bool,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let mut allowed = vec!["target", "uniform"];
    if vector {
        allowed.push("frame");
    }
    let mut assignments = BTreeSet::new();
    let mut source_blocks = BTreeSet::new();
    for statement in body {
        match &statement.kind {
            CheckedFeaStatementKind::Assignment {
                target, relation, ..
            } => {
                let Some(name) = target.first().filter(|_| target.len() == 1) else {
                    diagnostic(
                        declaration,
                        statement.span,
                        "initial-condition member names cannot be qualified",
                        diagnostics,
                    );
                    continue;
                };
                if *relation != FeaAssignmentRelation::Equal {
                    diagnostic(
                        declaration,
                        statement.span,
                        "initial-condition members require `=`",
                        diagnostics,
                    );
                } else if !allowed.contains(&name.as_str()) {
                    diagnostic(
                        declaration,
                        statement.span,
                        format!("unsupported initial-condition member `{name}`"),
                        diagnostics,
                    );
                } else if !assignments.insert(name.as_str()) {
                    diagnostic(
                        declaration,
                        statement.span,
                        format!("duplicate initial-condition member `{name}`"),
                        diagnostics,
                    );
                }
            }
            CheckedFeaStatementKind::Block { header, .. }
                if matches!(header.trim(), "sample" | "field_artifact") =>
            {
                if !source_blocks.insert(header.trim()) {
                    diagnostic(
                        declaration,
                        statement.span,
                        format!("duplicate initial-condition source `{}`", header.trim()),
                        diagnostics,
                    );
                }
            }
            _ => diagnostic(
                declaration,
                statement.span,
                "initial conditions accept assignments plus `sample` or `field_artifact` sources",
                diagnostics,
            ),
        }
    }
}
