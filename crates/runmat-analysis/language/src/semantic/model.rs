use super::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatementKind,
    FeaTopologyRole, FeaType,
};
use crate::{
    BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey,
    FeaSymbolKind, FeaSymbolOrigin, LogicalSourcePath,
};
use runmat_analysis_core::{
    AuthoredLayupSource, AuthoredMaterialAssignment, AuthoredMaterialRef, AuthoredMaterialSource,
    AuthoredModel, AuthoredModelInclude, AuthoredModelInputKind, AuthoredModelInputRef,
    AuthoredModelRef, AuthoredOrientationSource, AuthoredPropertyAssignment,
    AuthoredPropertyRegionSource, AuthoredPropertyTarget, AuthoredSectionSource,
};
use runmat_geometry_core::{GeometryRegionRef, GeometryRegionRole};
use std::collections::{BTreeMap, BTreeSet};

mod properties;

pub fn lower_models(
    bound: &BoundFeaProgram,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel> {
    let mut result = BTreeMap::new();
    for (identity, declaration) in declarations {
        if identity.1.kind != FeaSymbolKind::Model {
            continue;
        }
        let initial_errors = diagnostics.len();
        let model = model(bound, declaration, diagnostics);
        if diagnostics.len() != initial_errors {
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
    validate_composition(&mut result, declarations, diagnostics);
    result
}

fn validate_composition(
    models: &mut BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel>,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let inventory = models
        .iter()
        .map(|(key, model)| {
            (
                AuthoredModelRef {
                    module_path: model.module_path.clone(),
                    model_id: model.model_id.clone(),
                },
                key.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let keys = models.keys().cloned().collect::<Vec<_>>();
    let mut invalid = BTreeSet::new();
    for key in keys {
        let mut stack = BTreeSet::new();
        let mut assignments = CompositionAssignments::default();
        if let Err(message) =
            collect_composition(&key, models, &inventory, &mut stack, &mut assignments)
        {
            invalid.insert(key.clone());
            if let Some(declaration) = declarations.get(&key) {
                diagnostic(declaration, declaration.span, message, diagnostics);
            }
        }
    }
    models.retain(|key, _| !invalid.contains(key));
}

fn collect_composition(
    key: &(LogicalSourcePath, FeaSymbolKey),
    models: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel>,
    inventory: &BTreeMap<AuthoredModelRef, (LogicalSourcePath, FeaSymbolKey)>,
    stack: &mut BTreeSet<(LogicalSourcePath, FeaSymbolKey)>,
    assignments: &mut CompositionAssignments,
) -> Result<(), String> {
    if !stack.insert(key.clone()) {
        return Err("model composition contains an include cycle".into());
    }
    let model = models
        .get(key)
        .ok_or_else(|| "included model is not available for composition".to_string())?;
    for include in &model.includes {
        let target = inventory.get(&include.model).ok_or_else(|| {
            format!(
                "included model `{}::{}` is not available",
                include.model.module_path, include.model.model_id
            )
        })?;
        let included = models
            .get(target)
            .ok_or_else(|| "included model is not available for composition".to_string())?;
        if !included.property_inputs.is_empty() {
            return Err(format!(
                "included model `{}` requires explicit input binding",
                included.model_id
            ));
        }
        collect_composition(target, models, inventory, stack, assignments)?;
    }
    for assignment in &model.material_assignments {
        if let Some(existing) = assignments.materials.get(&assignment.target) {
            if existing != &assignment.material {
                return Err("model composition assigns conflicting materials to one target".into());
            }
        } else {
            assignments
                .materials
                .insert(assignment.target.clone(), assignment.material.clone());
        }
        if let AuthoredPropertyTarget::Shell(source) = &assignment.target {
            if assignments.layups.contains_key(source) {
                return Err(
                    "model composition gives a shell both homogeneous and laminate ownership"
                        .into(),
                );
            }
        }
    }
    for assignment in &model.property_assignments {
        match assignment {
            AuthoredPropertyAssignment::Section { section, target } => {
                insert_property(&mut assignments.sections, target, section, "section")?;
            }
            AuthoredPropertyAssignment::Layup { layup, target } => {
                insert_property(&mut assignments.layups, target, layup, "layup")?;
                if assignments
                    .materials
                    .contains_key(&AuthoredPropertyTarget::Shell(target.clone()))
                {
                    return Err(
                        "model composition gives a shell both homogeneous and laminate ownership"
                            .into(),
                    );
                }
            }
            AuthoredPropertyAssignment::Orientation {
                orientation,
                target,
            } => {
                insert_property(
                    &mut assignments.orientations,
                    target,
                    orientation,
                    "orientation",
                )?;
            }
        }
    }
    stack.remove(key);
    Ok(())
}

#[derive(Default)]
struct CompositionAssignments {
    materials: BTreeMap<AuthoredPropertyTarget, AuthoredMaterialSource>,
    sections: BTreeMap<AuthoredPropertyTarget, AuthoredSectionSource>,
    layups: BTreeMap<AuthoredPropertyRegionSource, AuthoredLayupSource>,
    orientations: BTreeMap<AuthoredPropertyTarget, AuthoredOrientationSource>,
}

fn insert_property<K, V>(
    assignments: &mut BTreeMap<K, V>,
    target: &K,
    value: &V,
    label: &str,
) -> Result<(), String>
where
    K: Ord + Clone,
    V: PartialEq + Clone,
{
    if let Some(existing) = assignments.get(target) {
        if existing != value {
            return Err(format!(
                "model composition assigns conflicting {label} definitions to one target"
            ));
        }
    } else {
        assignments.insert(target.clone(), value.clone());
    }
    Ok(())
}

fn model(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> AuthoredModel {
    let property_inputs = declaration
        .inputs
        .values()
        .filter_map(|input| {
            let kind = match &input.value_type {
                FeaType::Declaration(FeaSymbolKind::Material) => AuthoredModelInputKind::Material,
                FeaType::Region(role) => AuthoredModelInputKind::Region(geometry_role(*role)),
                _ => return None,
            };
            Some(AuthoredModelInputRef {
                input_id: input.name.clone(),
                kind,
            })
        })
        .collect();
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
    AuthoredModel {
        module_path: declaration.module.as_str().to_string(),
        model_id: declaration.key.name.clone(),
        property_inputs,
        includes,
        sections: lowered_properties.sections,
        layups: lowered_properties.layups,
        orientations: lowered_properties.orientations,
        material_assignments,
        property_assignments: lowered_properties.assignments,
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
                kind: AuthoredModelInputKind::Material,
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
            property_target_for_role(
                *role,
                AuthoredPropertyRegionSource::ModelInput(AuthoredModelInputRef {
                    input_id: name.clone(),
                    kind: AuthoredModelInputKind::Region(geometry_role(*role)),
                }),
                declaration,
                expression,
                diagnostics,
            )
        }
        [namespace, geometry, region_namespace, region]
            if namespace == "geometry" && region_namespace == "region" =>
        {
            let FeaType::Region(role) = &expression.value_type else {
                invalid_assignment(declaration, expression, "geometry region", diagnostics);
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
            property_target_for_role(
                *role,
                AuthoredPropertyRegionSource::Geometry(GeometryRegionRef {
                    module_path: reference.0,
                    geometry_id: reference.1,
                    region_id: region.clone(),
                    role: geometry_role(*role),
                }),
                declaration,
                expression,
                diagnostics,
            )
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
) -> Option<AuthoredPropertyRegionSource> {
    match property_target(bound, declaration, expression, diagnostics)? {
        AuthoredPropertyTarget::Beam(source)
        | AuthoredPropertyTarget::Shell(source)
        | AuthoredPropertyTarget::Solid(source)
        | AuthoredPropertyTarget::Cell(source) => Some(source),
    }
}

fn property_target_for_role(
    role: FeaTopologyRole,
    source: AuthoredPropertyRegionSource,
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
        format!("material assignment requires a {expected}"),
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
