use runmat_analysis_language::{
    CheckedFeaStatement, CheckedFeaStatementKind, FeaDiagnosticLocation, FeaModelDefinitionKind,
    FeaSymbolKey, LogicalSourcePath,
};
use serde::{Deserialize, Serialize};

use super::CheckedFeaSource;

/// Source navigation metadata kept beside, rather than embedded into, canonical FEA objects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaSourceIndex {
    pub entries: Vec<FeaSourceIndexEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaSourceIndexEntry {
    pub identity: FeaSourceIdentity,
    pub location: FeaDiagnosticLocation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeaSourceIdentity {
    Declaration {
        module_path: LogicalSourcePath,
        symbol: FeaSymbolKey,
    },
    ModelMember {
        module_path: LogicalSourcePath,
        model_id: String,
        member_kind: FeaModelMemberKind,
        member_id: String,
    },
    Function {
        module_path: LogicalSourcePath,
        function_id: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaModelMemberKind {
    Section,
    Layup,
    Orientation,
    Constraint,
    Load,
    LoadCase,
    Interface,
    Connector,
    Excitation,
    ResponseSpectrum,
    RandomExcitation,
    InitialCondition,
    Signal,
}

impl CheckedFeaSource {
    pub fn source_index(&self) -> FeaSourceIndex {
        let mut entries = Vec::new();
        for ((module_path, symbol), declaration) in &self.program.declarations {
            entries.push(FeaSourceIndexEntry {
                identity: FeaSourceIdentity::Declaration {
                    module_path: module_path.clone(),
                    symbol: symbol.clone(),
                },
                location: FeaDiagnosticLocation {
                    path: declaration.module.clone(),
                    span: declaration.span,
                },
            });
            if symbol.kind == runmat_analysis_language::FeaSymbolKind::Model {
                collect_model_members(
                    &mut entries,
                    module_path,
                    &symbol.name,
                    &declaration.statements,
                );
            }
        }
        for facts in self.program.bound.functions.values() {
            entries.push(FeaSourceIndexEntry {
                identity: FeaSourceIdentity::Function {
                    module_path: facts.module_path.clone(),
                    function_id: facts.symbol.as_str().to_string(),
                },
                location: FeaDiagnosticLocation {
                    path: facts.module_path.clone(),
                    span: facts.declaration_span,
                },
            });
        }
        entries.sort_by(|left, right| {
            source_identity_key(&left.identity).cmp(&source_identity_key(&right.identity))
        });
        FeaSourceIndex { entries }
    }
}

fn collect_model_members(
    entries: &mut Vec<FeaSourceIndexEntry>,
    module_path: &LogicalSourcePath,
    model_id: &str,
    statements: &[CheckedFeaStatement],
) {
    for statement in statements {
        let member = match &statement.kind {
            CheckedFeaStatementKind::ModelDefinition {
                kind,
                name,
                statements,
            } => {
                collect_model_members(entries, module_path, model_id, statements);
                Some((model_member_kind(*kind), name.as_str()))
            }
            CheckedFeaStatementKind::SignalDefinition {
                name, statements, ..
            } => {
                collect_model_members(entries, module_path, model_id, statements);
                Some((FeaModelMemberKind::Signal, name.as_str()))
            }
            _ => None,
        };
        if let Some((member_kind, member_id)) = member {
            entries.push(FeaSourceIndexEntry {
                identity: FeaSourceIdentity::ModelMember {
                    module_path: module_path.clone(),
                    model_id: model_id.to_string(),
                    member_kind,
                    member_id: member_id.to_string(),
                },
                location: FeaDiagnosticLocation {
                    path: module_path.clone(),
                    span: statement.span,
                },
            });
        }
    }
}

fn model_member_kind(kind: FeaModelDefinitionKind) -> FeaModelMemberKind {
    match kind {
        FeaModelDefinitionKind::Section => FeaModelMemberKind::Section,
        FeaModelDefinitionKind::Layup => FeaModelMemberKind::Layup,
        FeaModelDefinitionKind::Orientation => FeaModelMemberKind::Orientation,
        FeaModelDefinitionKind::Constraint => FeaModelMemberKind::Constraint,
        FeaModelDefinitionKind::Load => FeaModelMemberKind::Load,
        FeaModelDefinitionKind::LoadCase => FeaModelMemberKind::LoadCase,
        FeaModelDefinitionKind::Interface => FeaModelMemberKind::Interface,
        FeaModelDefinitionKind::Connector => FeaModelMemberKind::Connector,
        FeaModelDefinitionKind::Excitation => FeaModelMemberKind::Excitation,
        FeaModelDefinitionKind::ResponseSpectrum => FeaModelMemberKind::ResponseSpectrum,
        FeaModelDefinitionKind::RandomExcitation => FeaModelMemberKind::RandomExcitation,
        FeaModelDefinitionKind::Initial => FeaModelMemberKind::InitialCondition,
    }
}

fn source_identity_key(identity: &FeaSourceIdentity) -> String {
    match identity {
        FeaSourceIdentity::Declaration {
            module_path,
            symbol,
        } => format!(
            "0\0{}\0{}\0{}",
            module_path.as_str(),
            symbol.kind.namespace(),
            symbol.name
        ),
        FeaSourceIdentity::ModelMember {
            module_path,
            model_id,
            member_kind,
            member_id,
        } => format!(
            "1\0{}\0{}\0{member_kind:?}\0{}",
            module_path.as_str(),
            model_id,
            member_id
        ),
        FeaSourceIdentity::Function {
            module_path,
            function_id,
        } => format!("2\0{}\0{}", module_path.as_str(), function_id),
    }
}
