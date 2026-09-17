use std::collections::BTreeMap;

use crate::{
    BoundFeaProgram, CheckedCallArgument, CheckedExpression, CheckedExpressionKind,
    CheckedFeaDeclaration, FeaDiagnostic, FeaSymbolKey, FeaSymbolKind, FeaSymbolOrigin,
    LogicalSourcePath,
};
use runmat_analysis_core::{
    AuthoredMeshRef, AuthoredModel, AuthoredModelArgument, AuthoredModelArgumentValue,
    AuthoredModelInstance, AuthoredModelRef, AuthoredStudy, AuthoredStudyInputRef,
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
    let geometry = declaration_binding(
        context.bound,
        declaration,
        "geometry",
        FeaSymbolKind::Geometry,
        diagnostics,
    )?;
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

    let (model_key, model_expression) = model_binding(context.bound, declaration, diagnostics)?;
    let model = context.models.get(&model_key).or_else(|| {
        lower::error(
            declaration,
            model_expression.span,
            "study model is not available as a valid typed model",
            diagnostics,
        );
        None
    })?;
    let inputs = super::super::input::lower_definitions(context.bound, declaration, diagnostics);
    let arguments = model_arguments(
        context.bound,
        declaration,
        model_expression,
        model,
        diagnostics,
    );
    let sequence = lower::lower_study(declaration, model, diagnostics)?;
    if diagnostics.len() != initial_errors {
        return None;
    }
    let study = AuthoredStudy {
        module_path: declaration.module.as_str().to_string(),
        study_id: declaration.key.name.clone(),
        geometry: runmat_geometry_core::GeometryDefinitionRef {
            module_path: geometry_definition.module_path.clone(),
            geometry_id: geometry_definition.geometry_id.clone(),
        },
        mesh: AuthoredMeshRef {
            module_path: mesh_definition.module_path.clone(),
            mesh_id: mesh_definition.mesh_id.clone(),
        },
        model: AuthoredModelInstance {
            model: AuthoredModelRef {
                module_path: model.module_path.clone(),
                model_id: model.model_id.clone(),
            },
            arguments,
        },
        inputs,
        sequence,
    };
    if let Err(error) = study.validate(model) {
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
        .or_else(|| missing_binding(declaration, member, diagnostics))
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
        .or_else(|| missing_binding(declaration, "model", diagnostics))?;
    Some((key, expression))
}

fn symbol_key(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    path: &[String],
    kind: FeaSymbolKind,
) -> Option<(LogicalSourcePath, FeaSymbolKey)> {
    let [namespace, name] = path else {
        return None;
    };
    if namespace != namespace_for(kind) {
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
            let CheckedCallArgument::Named { name, value } = argument else {
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

fn namespace_for(kind: FeaSymbolKind) -> &'static str {
    match kind {
        FeaSymbolKind::Geometry => "geometry",
        FeaSymbolKind::Mesh => "mesh",
        FeaSymbolKind::Model => "model",
        _ => "",
    }
}
