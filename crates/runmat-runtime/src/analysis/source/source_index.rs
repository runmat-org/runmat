use runmat_analysis_language::{
    CheckedFeaStatement, CheckedFeaStatementKind, FeaDiagnosticLocation, FeaModelDefinitionKind,
    LogicalSourcePath,
};

use super::CheckedFeaSource;

pub use runmat_fea_source::{
    FeaModelMemberKind, FeaSourceIdentity, FeaSourceIndex, FeaSourceIndexEntry, FeaStudyMemberKind,
};

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
            if symbol.kind == runmat_analysis_language::FeaSymbolKind::Study {
                collect_study_members(
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

fn collect_study_members(
    entries: &mut Vec<FeaSourceIndexEntry>,
    module_path: &LogicalSourcePath,
    study_id: &str,
    statements: &[CheckedFeaStatement],
) {
    for statement in statements {
        let (member_kind, member_id) = match &statement.kind {
            CheckedFeaStatementKind::Output { name, .. }
            | CheckedFeaStatementKind::OutputDefinition { name, .. } => {
                (FeaStudyMemberKind::Output, name)
            }
            CheckedFeaStatementKind::AssessmentDefinition { name, .. } => {
                (FeaStudyMemberKind::Assessment, name)
            }
            _ => continue,
        };
        entries.push(FeaSourceIndexEntry {
            identity: FeaSourceIdentity::StudyMember {
                module_path: module_path.clone(),
                study_id: study_id.into(),
                member_kind,
                member_id: member_id.clone(),
            },
            location: FeaDiagnosticLocation {
                path: module_path.clone(),
                span: statement.span,
            },
        });
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
        FeaSourceIdentity::StudyMember {
            module_path,
            study_id,
            member_kind,
            member_id,
        } => format!(
            "3\0{}\0{}\0{member_kind:?}\0{}",
            module_path.as_str(),
            study_id,
            member_id
        ),
    }
}
