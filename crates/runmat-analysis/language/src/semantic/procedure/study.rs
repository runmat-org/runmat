use std::collections::BTreeMap;

use crate::{
    BoundFeaProgram, CheckedCallArgument, CheckedExpression, CheckedExpressionKind,
    CheckedFeaDeclaration, FeaDiagnostic, FeaSymbolKey, FeaSymbolKind, FeaSymbolOrigin,
    LogicalSourcePath,
};
use runmat_analysis_core::{
    geometry_parameter_input_type, AuthoredGeometryArgument, AuthoredGeometryArgumentValue,
    AuthoredGeometryInstance, AuthoredMeshRef, AuthoredModel, AuthoredModelArgument,
    AuthoredModelArgumentValue, AuthoredModelInstance, AuthoredModelRef, AuthoredStudy,
    AuthoredStudyInputRef,
};
use runmat_geometry_core::GeometryDefinition;
use runmat_meshing_core::MeshDefinition;

use super::{lower, values};

pub(super) struct StudyLowering<'a> {
    pub bound: &'a BoundFeaProgram,
    pub models: &'a BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel>,
    pub geometries: &'a BTreeMap<(LogicalSourcePath, FeaSymbolKey), GeometryDefinition>,
    pub meshes: &'a BTreeMap<(LogicalSourcePath, FeaSymbolKey), MeshDefinition>,
}

pub(super) fn lower(
    context: &StudyLowering<'_>,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredStudy> {
    let initial_errors = diagnostics.len();
    let (geometry, geometry_expression) =
        geometry_binding(context.bound, declaration, diagnostics)?;
    let geometry_definition = context.geometries.get(&geometry)?;
    let mesh = declaration_binding(
        context.bound,
        declaration,
        "mesh",
        FeaSymbolKind::Mesh,
        diagnostics,
    )?;
    let mesh_definition = context.meshes.get(&mesh)?;
    if mesh_definition.geometry.module_path != geometry_definition.module_path
        || mesh_definition.geometry.geometry_id != geometry_definition.geometry_id
    {
        lower::error(
            declaration,
            declaration.span,
            "study mesh must be authored for the study geometry",
            diagnostics,
        );
    }

    // A study on a cross-section takes no model: the section carries its dielectrics.
    let section = matches!(
        geometry_definition.source,
        runmat_geometry_core::GeometryDefinitionSource::CrossSection(_)
    );
    let empty;
    let bound_model = if section && values::assignment(&declaration.statements, "model").is_none() {
        None
    } else {
        let (model_key, model_expression) = model_binding(context.bound, declaration, diagnostics)?;
        // A model that failed to lower has already reported its own located diagnostics;
        // repeating that failure at every dependent study would only add cascade errors.
        Some((context.models.get(&model_key)?, model_expression))
    };
    let model = match bound_model {
        Some((model, _)) => model,
        None => {
            empty = AuthoredModel::empty(declaration.module.as_str(), &declaration.key.name);
            &empty
        }
    };
    let inputs = super::super::input::lower_definitions(context.bound, declaration, diagnostics);
    let geometry_arguments = geometry_arguments(
        context.bound,
        declaration,
        geometry_expression,
        geometry_definition,
        diagnostics,
    );
    let arguments = bound_model.map(|(model, expression)| {
        model_arguments(context.bound, declaration, expression, model, diagnostics)
    });
    let model_closure = match bound_model {
        Some((model, expression)) => {
            let closure = model_closure(context.models, model)?;
            require_sections_for_structural_domains(
                declaration,
                expression.span,
                &closure,
                diagnostics,
            );
            closure
        }
        None => vec![model],
    };
    let mut sequence = lower::lower_study(declaration, &model_closure, diagnostics)?;
    let outputs = super::output::lower_outputs(context.bound, declaration, diagnostics);
    reject_formulation_setting(declaration, diagnostics);
    let assessments = super::assessment::lower_assessments(context.bound, declaration, diagnostics);
    match runmat_analysis_core::attach_authored_outputs(&outputs, &mut sequence) {
        Ok(()) => {}
        Err(error) => lower::error(
            declaration,
            declaration.span,
            error.to_string(),
            diagnostics,
        ),
    }
    if diagnostics.len() != initial_errors {
        return None;
    }
    let study = AuthoredStudy {
        module_path: declaration.module.as_str().to_string(),
        study_id: declaration.key.name.clone(),
        geometry: AuthoredGeometryInstance {
            definition: runmat_geometry_core::GeometryDefinitionRef {
                module_path: geometry_definition.module_path.clone(),
                geometry_id: geometry_definition.geometry_id.clone(),
            },
            arguments: geometry_arguments,
        },
        mesh: AuthoredMeshRef {
            module_path: mesh_definition.module_path.clone(),
            mesh_id: mesh_definition.mesh_id.clone(),
        },
        model: arguments.map(|arguments| AuthoredModelInstance {
            model: AuthoredModelRef {
                module_path: model.module_path.clone(),
                model_id: model.model_id.clone(),
            },
            arguments,
        }),
        inputs,
        outputs,
        assessments,
        sequence,
    };
    if let Err(error) = study.validate(model, geometry_definition) {
        lower::error(
            declaration,
            declaration.span,
            error.to_string(),
            diagnostics,
        );
        return None;
    }
    Some(study)
}

/// `formulation` is not a study setting: every registered formulation runs on one path. A stale
/// `formulation = ...` line is rejected so the authored file states only what the study does.
fn reject_formulation_setting(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    for statement in &declaration.statements {
        let target = match &statement.kind {
            crate::CheckedFeaStatementKind::Assignment { target, .. }
            | crate::CheckedFeaStatementKind::Requirement { target, .. } => target,
            _ => continue,
        };
        if target.as_slice() == ["formulation"] {
            lower::error(
                declaration,
                statement.span,
                "`formulation` is not a study setting; remove this line",
                diagnostics,
            );
        }
    }
}

/// A material assigned to a face or an edge declares a shell or beam domain, which has no
/// thickness or cross-section of its own. Without a section (or a layup for a shell) on the same
/// region, the study would solve a body that has no material where the author meant one.
fn require_sections_for_structural_domains(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    closure: &[&AuthoredModel],
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    use runmat_analysis_core::{
        AuthoredPropertyAssignment, AuthoredPropertyTarget, AuthoredRegionSource,
    };
    let has_section = |target: &AuthoredPropertyTarget| {
        closure.iter().any(|model| {
            model
                .property_assignments
                .iter()
                .any(|assignment| match assignment {
                    AuthoredPropertyAssignment::Section {
                        target: section, ..
                    } => section == target,
                    AuthoredPropertyAssignment::Layup { target: layup, .. } => {
                        matches!(target, AuthoredPropertyTarget::Shell(shell) if shell == layup)
                    }
                    AuthoredPropertyAssignment::Orientation { .. } => false,
                })
        })
    };
    for model in closure {
        for assignment in &model.material_assignments {
            let (kind, needs) = match &assignment.target {
                AuthoredPropertyTarget::Shell(_) => ("face", "a shell section or layup"),
                AuthoredPropertyTarget::Beam(_) => ("edge", "a beam section"),
                AuthoredPropertyTarget::Solid(_) | AuthoredPropertyTarget::Cell(_) => continue,
            };
            if has_section(&assignment.target) {
                continue;
            }
            let (AuthoredPropertyTarget::Shell(source) | AuthoredPropertyTarget::Beam(source)) =
                &assignment.target
            else {
                continue;
            };
            let region = match source {
                AuthoredRegionSource::Geometry(region) => {
                    format!(
                        "geometry.{}.region.{}",
                        region.geometry_id, region.region_id
                    )
                }
                AuthoredRegionSource::ModelInput(input) => format!("input.{}", input.input_id),
            };
            lower::error(
                declaration,
                span,
                format!(
                    "model `{}` assigns a material to {kind} region `{region}`, which needs {needs}; assign the material to a solid region or add the section",
                    model.model_id
                ),
                diagnostics,
            );
        }
    }
}

fn model_closure<'a>(
    models: &'a BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel>,
    root: &'a AuthoredModel,
) -> Option<Vec<&'a AuthoredModel>> {
    fn collect<'a>(
        models: &'a BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel>,
        model: &'a AuthoredModel,
        collected: &mut Vec<&'a AuthoredModel>,
    ) -> Option<()> {
        for include in &model.includes {
            let included = models.values().find(|candidate| {
                candidate.module_path == include.model.module_path
                    && candidate.model_id == include.model.model_id
            })?;
            collect(models, included, collected)?;
        }
        if !collected.iter().any(|candidate| {
            candidate.module_path == model.module_path && candidate.model_id == model.model_id
        }) {
            collected.push(model);
        }
        Some(())
    }

    let mut collected = Vec::new();
    collect(models, root, &mut collected)?;
    Some(collected)
}

fn geometry_binding<'a>(
    bound: &BoundFeaProgram,
    declaration: &'a CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<((LogicalSourcePath, FeaSymbolKey), &'a CheckedExpression)> {
    let Some(expression) = values::assignment(&declaration.statements, "geometry") else {
        return missing_binding(declaration, "geometry", diagnostics);
    };
    let path = match &expression.kind {
        CheckedExpressionKind::Reference(path)
        | CheckedExpressionKind::FunctionCall { function: path, .. } => path,
        _ => return missing_binding(declaration, "geometry", diagnostics),
    };
    let key = symbol_key(bound, declaration, path, FeaSymbolKind::Geometry)
        .or_else(|| unresolved_binding(declaration, expression, "geometry", diagnostics))?;
    Some((key, expression))
}

fn geometry_arguments(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    geometry: &GeometryDefinition,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<AuthoredGeometryArgument> {
    let CheckedExpressionKind::FunctionCall { arguments, .. } = &expression.kind else {
        return Vec::new();
    };
    let parameters = geometry
        .parameters
        .iter()
        .map(|parameter| (parameter.parameter_id.as_str(), parameter))
        .collect::<BTreeMap<_, _>>();
    arguments
        .iter()
        .filter_map(|argument| {
            let CheckedCallArgument::Named { name, value, .. } = argument else {
                lower::error(
                    declaration,
                    expression.span,
                    "geometry bindings in a study must use named arguments",
                    diagnostics,
                );
                return None;
            };
            let Some(parameter) = parameters.get(name.as_str()) else {
                lower::error(
                    declaration,
                    value.span,
                    format!("geometry has no input `{name}`"),
                    diagnostics,
                );
                return None;
            };
            let Some(expected) = geometry_parameter_input_type(parameter) else {
                lower::error(
                    declaration,
                    value.span,
                    format!("geometry input `{name}` cannot be bound from a study"),
                    diagnostics,
                );
                return None;
            };
            let bound_value = match &value.kind {
                CheckedExpressionKind::Reference(path)
                    if matches!(path.as_slice(), [namespace, _] if namespace == "input") =>
                {
                    let input_id = path[1].clone();
                    let Some(input) = declaration.inputs.get(&input_id) else {
                        lower::error(
                            declaration,
                            value.span,
                            format!("study has no input `{input_id}`"),
                            diagnostics,
                        );
                        return None;
                    };
                    let Some(value_type) = super::super::input::authored_type(&input.value_type)
                    else {
                        lower::error(
                            declaration,
                            value.span,
                            format!("study input `{input_id}` has no canonical authored type"),
                            diagnostics,
                        );
                        return None;
                    };
                    AuthoredGeometryArgumentValue::StudyInput(AuthoredStudyInputRef {
                        input_id,
                        value_type,
                    })
                }
                _ => AuthoredGeometryArgumentValue::Constant(
                    super::super::input::lower_value(
                        bound,
                        declaration,
                        expected,
                        value,
                        diagnostics,
                    )?,
                ),
            };
            Some(AuthoredGeometryArgument {
                parameter_id: name.clone(),
                value: bound_value,
            })
        })
        .collect()
}

fn declaration_binding(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    member: &str,
    kind: FeaSymbolKind,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(LogicalSourcePath, FeaSymbolKey)> {
    let Some(expression) = values::assignment(&declaration.statements, member) else {
        return missing_binding(declaration, member, diagnostics);
    };
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        return missing_binding(declaration, member, diagnostics);
    };
    symbol_key(bound, declaration, path, kind)
        .or_else(|| unresolved_binding(declaration, expression, member, diagnostics))
}

fn model_binding<'a>(
    bound: &BoundFeaProgram,
    declaration: &'a CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<((LogicalSourcePath, FeaSymbolKey), &'a CheckedExpression)> {
    let Some(expression) = values::assignment(&declaration.statements, "model") else {
        return missing_binding(declaration, "model", diagnostics);
    };
    let path = match &expression.kind {
        CheckedExpressionKind::Reference(path)
        | CheckedExpressionKind::FunctionCall { function: path, .. } => path,
        _ => return missing_binding(declaration, "model", diagnostics),
    };
    let key = symbol_key(bound, declaration, path, FeaSymbolKind::Model)
        .or_else(|| unresolved_binding(declaration, expression, "model", diagnostics))?;
    Some((key, expression))
}

pub(crate) fn symbol_key(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    path: &[String],
    kind: FeaSymbolKind,
) -> Option<(LogicalSourcePath, FeaSymbolKey)> {
    let [namespace, name] = path else {
        return None;
    };
    if namespace != kind.namespace() {
        return None;
    }
    let key = FeaSymbolKey {
        kind,
        name: name.clone(),
    };
    let symbol = bound
        .symbols
        .get(&declaration.module)
        .and_then(|symbols| symbols.get(&key))?;
    match &symbol.origin {
        FeaSymbolOrigin::Local => Some((declaration.module.clone(), key)),
        FeaSymbolOrigin::Import {
            target_module,
            target,
        } => Some((target_module.clone(), target.clone())),
        FeaSymbolOrigin::Function => None,
    }
}

fn model_arguments(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    model: &AuthoredModel,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<AuthoredModelArgument> {
    let CheckedExpressionKind::FunctionCall { arguments, .. } = &expression.kind else {
        return Vec::new();
    };
    let definitions = model
        .inputs
        .iter()
        .map(|input| (input.input_id.as_str(), input))
        .collect::<BTreeMap<_, _>>();
    arguments
        .iter()
        .filter_map(|argument| {
            let CheckedCallArgument::Named { name, value, .. } = argument else {
                lower::error(
                    declaration,
                    expression.span,
                    "model bindings in a study must use named arguments",
                    diagnostics,
                );
                return None;
            };
            let Some(definition) = definitions.get(name.as_str()) else {
                lower::error(
                    declaration,
                    value.span,
                    format!("model has no input `{name}`"),
                    diagnostics,
                );
                return None;
            };
            let bound_value = match &value.kind {
                CheckedExpressionKind::Reference(path)
                    if matches!(path.as_slice(), [namespace, _] if namespace == "input") =>
                {
                    let input_id = path[1].clone();
                    let Some(input) = declaration.inputs.get(&input_id) else {
                        lower::error(
                            declaration,
                            value.span,
                            format!("study has no input `{input_id}`"),
                            diagnostics,
                        );
                        return None;
                    };
                    let Some(value_type) = super::super::input::authored_type(&input.value_type)
                    else {
                        lower::error(
                            declaration,
                            value.span,
                            format!("study input `{input_id}` has no canonical authored type"),
                            diagnostics,
                        );
                        return None;
                    };
                    AuthoredModelArgumentValue::StudyInput(AuthoredStudyInputRef {
                        input_id,
                        value_type,
                    })
                }
                _ => AuthoredModelArgumentValue::Constant(
                    super::super::input::lower_value(
                        bound,
                        declaration,
                        definition.value_type,
                        value,
                        diagnostics,
                    )?,
                ),
            };
            Some(AuthoredModelArgument {
                input_id: name.clone(),
                value: bound_value,
            })
        })
        .collect()
}

/// A binding whose reference already failed to resolve was reported where it is written.
fn unresolved_binding<T>(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    member: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<T> {
    if expression.value_type == crate::FeaType::Error {
        return None;
    }
    missing_binding(declaration, member, diagnostics)
}

fn missing_binding<T>(
    declaration: &CheckedFeaDeclaration,
    member: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<T> {
    lower::error(
        declaration,
        declaration.span,
        format!("study requires one available typed `{member}` binding"),
        diagnostics,
    );
    None
}
