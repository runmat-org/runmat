use std::collections::BTreeMap;

use runmat_analysis_language::{
    CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind, FeaDiagnosticLocation,
    FeaModelDefinitionKind, FeaSymbolKey, FeaSymbolKind, LogicalSourcePath,
};

#[derive(Debug, Default)]
pub(super) struct PresentationSourceMap {
    declarations: BTreeMap<(LogicalSourcePath, FeaSymbolKey), FeaDiagnosticLocation>,
    model_objects: BTreeMap<(LogicalSourcePath, String, String, String), FeaDiagnosticLocation>,
}

impl PresentationSourceMap {
    pub(super) fn from_declarations<'a>(
        declarations: impl Iterator<
            Item = (
                &'a (LogicalSourcePath, FeaSymbolKey),
                &'a CheckedFeaDeclaration,
            ),
        >,
    ) -> Self {
        let mut result = Self::default();
        for (identity, declaration) in declarations {
            result.declarations.insert(
                identity.clone(),
                FeaDiagnosticLocation {
                    path: declaration.module.clone(),
                    span: declaration.span,
                },
            );
            if identity.1.kind == FeaSymbolKind::Model {
                collect_model_objects(
                    &mut result.model_objects,
                    &declaration.module,
                    &identity.1.name,
                    &declaration.statements,
                );
            }
        }
        result
    }

    pub(super) fn declaration(
        &self,
        module: &LogicalSourcePath,
        kind: FeaSymbolKind,
        name: &str,
    ) -> Option<FeaDiagnosticLocation> {
        self.declarations
            .get(&(
                module.clone(),
                FeaSymbolKey {
                    kind,
                    name: name.to_string(),
                },
            ))
            .cloned()
    }

    pub(super) fn model_object(
        &self,
        module: &str,
        model_id: &str,
        kind: &str,
        name: &str,
    ) -> Option<FeaDiagnosticLocation> {
        let module = LogicalSourcePath::new(module).ok()?;
        self.model_objects
            .get(&(
                module,
                model_id.to_string(),
                kind.to_string(),
                name.to_string(),
            ))
            .cloned()
    }
}

fn collect_model_objects(
    output: &mut BTreeMap<(LogicalSourcePath, String, String, String), FeaDiagnosticLocation>,
    module: &LogicalSourcePath,
    model_id: &str,
    statements: &[CheckedFeaStatement],
) {
    for statement in statements {
        match &statement.kind {
            CheckedFeaStatementKind::ModelDefinition {
                kind,
                name,
                statements,
            } => {
                output.insert(
                    (
                        module.clone(),
                        model_id.to_string(),
                        model_definition_kind(*kind).to_string(),
                        name.clone(),
                    ),
                    FeaDiagnosticLocation {
                        path: module.clone(),
                        span: statement.span,
                    },
                );
                collect_model_objects(output, module, model_id, statements);
            }
            CheckedFeaStatementKind::SignalDefinition {
                name, statements, ..
            } => {
                output.insert(
                    (
                        module.clone(),
                        model_id.to_string(),
                        "signal".to_string(),
                        name.clone(),
                    ),
                    FeaDiagnosticLocation {
                        path: module.clone(),
                        span: statement.span,
                    },
                );
                collect_model_objects(output, module, model_id, statements);
            }
            _ => {}
        }
    }
}

fn model_definition_kind(kind: FeaModelDefinitionKind) -> &'static str {
    match kind {
        FeaModelDefinitionKind::Section => "section",
        FeaModelDefinitionKind::Layup => "layup",
        FeaModelDefinitionKind::Orientation => "orientation",
        FeaModelDefinitionKind::Constraint => "constraint",
        FeaModelDefinitionKind::Load => "load",
        FeaModelDefinitionKind::LoadCase => "load_case",
        FeaModelDefinitionKind::Interface => "interface",
        FeaModelDefinitionKind::Connector => "connector",
        FeaModelDefinitionKind::Excitation => "excitation",
        FeaModelDefinitionKind::ResponseSpectrum => "response_spectrum",
        FeaModelDefinitionKind::RandomExcitation => "random_excitation",
        FeaModelDefinitionKind::Initial => "initial",
    }
}
