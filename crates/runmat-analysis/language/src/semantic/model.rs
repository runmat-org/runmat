use super::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatementKind,
    FeaTopologyRole, FeaType,
};
use crate::{
    BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey,
    FeaSymbolKind, FeaSymbolOrigin, LogicalSourcePath,
};
use runmat_analysis_core::{
    AuthoredInputType, AuthoredMaterialAssignment, AuthoredMaterialRef, AuthoredMaterialSource,
    AuthoredModel, AuthoredModelInclude, AuthoredModelInputRef, AuthoredModelRef,
    AuthoredPropertyTarget, AuthoredRegionSource,
};
use runmat_geometry_core::{GeometryRegionRef, GeometryRegionRole};
use std::collections::BTreeMap;

mod composition;
mod constraints;
mod domains;
mod excitations;
mod initial_conditions;
mod physics;
mod properties;
mod signals;
mod spectral_inputs;
mod tables;

pub fn lower_models(
    bound: &BoundFeaProgram,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel> {
    let mut result = BTreeMap::new();
    let tables = tables::lower_impedance_tables(bound, declarations, diagnostics);
    for (identity, declaration) in declarations {
        if identity.1.kind != FeaSymbolKind::Model {
            continue;
        }
        let initial_errors = diagnostics.len();
        let model = model(bound, declaration, &tables, diagnostics);
        if diagnostics.len() != initial_errors {
            continue;
        }
        // Whole-model validation of a declaration whose source already failed to check (an
        // unresolved reference or an invalid unit) would only restate that failure.
        if has_reported_error(declaration, diagnostics) {
            continue;
        }
        match model.validate() {
            Ok(()) => {
                result.insert(identity.clone(), model);
            }
            Err(error) => diagnostic(
                declaration,
                declaration.span,
                error.to_string(),
                diagnostics,
            ),
        }
    }
    composition::validate(&mut result, declarations, diagnostics);
    result
}

fn has_reported_error(declaration: &CheckedFeaDeclaration, diagnostics: &[FeaDiagnostic]) -> bool {
    diagnostics.iter().any(|diagnostic| {
        diagnostic.severity == crate::FeaDiagnosticSeverity::Error
            && diagnostic.location.as_ref().is_some_and(|location| {
                location.path == declaration.module
                    && location.span.start >= declaration.span.start
                    && location.span.end <= declaration.span.end
            })
    })
}

fn model(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    tables: &tables::ImpedanceTables,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> AuthoredModel {
    let inputs = super::input::lower_definitions(bound, declaration, diagnostics);
    let includes = declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::Include(expression) = &statement.kind else {
                return None;
            };
            include(bound, declaration, expression, diagnostics)
        })
        .collect();
    let material_assignments = declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::PropertyAssignment { property, target } = &statement.kind
            else {
                return None;
            };
            if property.value_type != FeaType::Declaration(FeaSymbolKind::Material)
                && property.value_type != FeaType::Error
            {
                return None;
            }
            assignment(bound, declaration, property, target, diagnostics)
        })
        .collect();
    let lowered_properties = properties::lower(declaration, bound, diagnostics);
    let lowered_physics = physics::lower(declaration, bound, tables, diagnostics);
    let (thermal, thermo_mechanical, electromagnetic) = domains::lower(declaration, diagnostics);
    let mut signals = signals::lower(declaration, bound, diagnostics);
    let constraints = constraints::lower(declaration, bound, &signals, diagnostics);
    let lowered_excitations = excitations::lower(declaration, bound, &signals, diagnostics);
    let initial_conditions = initial_conditions::lower(declaration, bound, &signals, diagnostics);
    let spectral_inputs = spectral_inputs::lower(bound, declaration, diagnostics);
    signals.extend(lowered_excitations.signals);
    AuthoredModel {
        module_path: declaration.module.as_str().to_string(),
        model_id: declaration.key.name.clone(),
        inputs,
        includes,
        sections: lowered_properties.sections,
        layups: lowered_properties.layups,
        orientations: lowered_properties.orientations,
        material_assignments,
        property_assignments: lowered_properties.assignments,
        constraints,
        boundaries: lowered_physics.boundaries,
        connectors: lowered_physics.connectors,
        interfaces: lowered_physics.interfaces,
        signals,
        parameterized_signals: lowered_excitations.parameterized_signals,
        excitations: lowered_excitations.excitations,
        excitation_cases: lowered_excitations.cases,
        response_spectra: spectral_inputs.response_spectra,
        random_excitations: spectral_inputs.random_excitations,
        initial_conditions,
        thermal,
        electromagnetic,
        thermo_mechanical,
    }
}

fn include(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredModelInclude> {
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        diagnostic(
            declaration,
            expression.span,
            "model include requires a model reference",
            diagnostics,
        );
        return None;
    };
    let reference = resolve_declaration_ref(
        bound,
        declaration,
        path,
        FeaSymbolKind::Model,
        expression.span,
        diagnostics,
    )?;
    Some(AuthoredModelInclude {
        model: AuthoredModelRef {
            module_path: reference.0,
            model_id: reference.1,
        },
    })
}

fn assignment(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    material: &CheckedExpression,
    target: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredMaterialAssignment> {
    Some(AuthoredMaterialAssignment {
        material: material_source(bound, declaration, material, diagnostics)?,
        target: property_target(bound, declaration, target, diagnostics)?,
    })
}

pub(super) fn material_source(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredMaterialSource> {
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        invalid_assignment(declaration, expression, "material source", diagnostics);
        return None;
    };
    match path.as_slice() {
        [namespace, _name] if namespace == "material" => {
            if expression.value_type == FeaType::Error {
                return None;
            }
            let reference = resolve_declaration_ref(
                bound,
                declaration,
                path,
                FeaSymbolKind::Material,
                expression.span,
                diagnostics,
            )?;
            Some(AuthoredMaterialSource::Declaration(AuthoredMaterialRef {
                module_path: reference.0,
                material_id: reference.1,
            }))
        }
        [namespace, name] if namespace == "input" => {
            let input = declaration.inputs.get(name)?;
            if input.value_type != FeaType::Declaration(FeaSymbolKind::Material) {
                invalid_assignment(declaration, expression, "material-typed input", diagnostics);
                return None;
            }
            Some(AuthoredMaterialSource::ModelInput(AuthoredModelInputRef {
                input_id: name.clone(),
                value_type: AuthoredInputType::Material,
            }))
        }
        _ => {
            invalid_assignment(declaration, expression, "material reference", diagnostics);
            None
        }
    }
}

pub(super) fn property_target(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredPropertyTarget> {
    let source = region_source(bound, declaration, expression, diagnostics)?;
    let role = region_source_role(&source)?;
    property_target_for_role(role, source, declaration, expression, diagnostics)
}

pub(super) fn region_source(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredRegionSource> {
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        invalid_assignment(declaration, expression, "region target", diagnostics);
        return None;
    };
    match path.as_slice() {
        [namespace, name] if namespace == "input" => {
            let input = declaration.inputs.get(name)?;
            let FeaType::Region(role) = &input.value_type else {
                invalid_assignment(declaration, expression, "region-typed input", diagnostics);
                return None;
            };
            Some(AuthoredRegionSource::ModelInput(AuthoredModelInputRef {
                input_id: name.clone(),
                value_type: AuthoredInputType::Region(geometry_role(*role)),
            }))
        }
        [namespace, geometry, region_namespace, region]
            if namespace == "geometry" && region_namespace == "region" =>
        {
            let FeaType::Region(role) = &expression.value_type else {
                if expression.value_type != FeaType::Error {
                    invalid_assignment(declaration, expression, "geometry region", diagnostics);
                }
                return None;
            };
            let reference = resolve_declaration_ref(
                bound,
                declaration,
                &[namespace.clone(), geometry.clone()],
                FeaSymbolKind::Geometry,
                expression.span,
                diagnostics,
            )?;
            Some(AuthoredRegionSource::Geometry(GeometryRegionRef {
                module_path: reference.0,
                geometry_id: reference.1,
                region_id: region.clone(),
                role: geometry_role(*role),
            }))
        }
        _ => {
            invalid_assignment(
                declaration,
                expression,
                "geometry or input region",
                diagnostics,
            );
            None
        }
    }
}

pub(super) fn property_region_source(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredRegionSource> {
    match property_target(bound, declaration, expression, diagnostics)? {
        AuthoredPropertyTarget::Beam(source)
        | AuthoredPropertyTarget::Shell(source)
        | AuthoredPropertyTarget::Solid(source)
        | AuthoredPropertyTarget::Cell(source) => Some(source),
    }
}

fn region_source_role(source: &AuthoredRegionSource) -> Option<FeaTopologyRole> {
    let role = match source {
        AuthoredRegionSource::Geometry(region) => region.role,
        AuthoredRegionSource::ModelInput(input) => match input.value_type {
            AuthoredInputType::Region(role) => role,
            _ => return None,
        },
    };
    Some(match role {
        GeometryRegionRole::Vertex => FeaTopologyRole::Vertex,
        GeometryRegionRole::Edge => FeaTopologyRole::Edge,
        GeometryRegionRole::Face => FeaTopologyRole::Face,
        GeometryRegionRole::Solid => FeaTopologyRole::Solid,
        GeometryRegionRole::Cell => FeaTopologyRole::Cell,
    })
}

fn property_target_for_role(
    role: FeaTopologyRole,
    source: AuthoredRegionSource,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredPropertyTarget> {
    Some(match role {
        FeaTopologyRole::Edge => AuthoredPropertyTarget::Beam(source),
        FeaTopologyRole::Face => AuthoredPropertyTarget::Shell(source),
        FeaTopologyRole::Solid => AuthoredPropertyTarget::Solid(source),
        FeaTopologyRole::Cell => AuthoredPropertyTarget::Cell(source),
        FeaTopologyRole::Vertex => {
            invalid_assignment(
                declaration,
                expression,
                "beam, shell, solid, or cell material domain",
                diagnostics,
            );
            return None;
        }
    })
}

fn resolve_declaration_ref(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    path: &[String],
    kind: FeaSymbolKind,
    span: runmat_types::Span,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(String, String)> {
    let [_, name] = path else {
        diagnostic(
            declaration,
            span,
            "invalid declaration reference",
            diagnostics,
        );
        return None;
    };
    let key = FeaSymbolKey {
        kind,
        name: name.clone(),
    };
    let Some(symbol) = bound
        .symbols
        .get(&declaration.module)
        .and_then(|symbols| symbols.get(&key))
    else {
        diagnostic(
            declaration,
            span,
            format!("unknown {} `{name}`", kind_name(kind)),
            diagnostics,
        );
        return None;
    };
    match &symbol.origin {
        FeaSymbolOrigin::Local => Some((declaration.module.as_str().to_string(), name.clone())),
        FeaSymbolOrigin::Import {
            target_module,
            target,
        } => Some((target_module.as_str().to_string(), target.name.clone())),
        FeaSymbolOrigin::Function => {
            diagnostic(
                declaration,
                span,
                "reference resolved to a function",
                diagnostics,
            );
            None
        }
    }
}

fn geometry_role(role: FeaTopologyRole) -> GeometryRegionRole {
    match role {
        FeaTopologyRole::Vertex => GeometryRegionRole::Vertex,
        FeaTopologyRole::Edge => GeometryRegionRole::Edge,
        FeaTopologyRole::Face => GeometryRegionRole::Face,
        FeaTopologyRole::Solid => GeometryRegionRole::Solid,
        FeaTopologyRole::Cell => GeometryRegionRole::Cell,
    }
}

fn kind_name(kind: FeaSymbolKind) -> &'static str {
    match kind {
        FeaSymbolKind::Geometry => "geometry",
        FeaSymbolKind::Material => "material",
        FeaSymbolKind::Model => "model",
        _ => "declaration",
    }
}

fn invalid_assignment(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    expected: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostic(
        declaration,
        expression.span,
        format!("model definition requires a {expected}"),
        diagnostics,
    );
}

pub(super) fn diagnostic(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.model",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    ));
}
