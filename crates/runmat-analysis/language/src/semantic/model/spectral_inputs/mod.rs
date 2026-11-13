use crate::{BoundFeaProgram, CheckedFeaDeclaration, CheckedFeaStatementKind, FeaDiagnostic};
use runmat_analysis_core::{RandomExcitationSpectrum, ResponseSpectrum};
use runmat_design_syntax::FeaModelDefinitionKind;

mod random;
mod response;
mod values;

pub(super) struct LoweredSpectralInputs {
    pub response_spectra: Vec<ResponseSpectrum>,
    pub random_excitations: Vec<RandomExcitationSpectrum>,
}

pub(super) fn lower(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> LoweredSpectralInputs {
    let mut response_spectra = Vec::new();
    let mut random_excitations = Vec::new();
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
            FeaModelDefinitionKind::ResponseSpectrum => {
                if let Some(value) =
                    response::lower(bound, declaration, name, statements, diagnostics)
                {
                    response_spectra.push(value);
                }
            }
            FeaModelDefinitionKind::RandomExcitation => {
                if let Some(value) =
                    random::lower(bound, declaration, name, statements, diagnostics)
                {
                    random_excitations.push(value);
                }
            }
            _ => {}
        }
    }
    LoweredSpectralInputs {
        response_spectra,
        random_excitations,
    }
}

fn authored_digest(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    kind: &str,
    name: &str,
) -> Option<String> {
    use sha2::{Digest, Sha256};

    let source = bound
        .graph
        .modules
        .get(&declaration.module)?
        .source
        .source_digest
        .0;
    let mut digest = Sha256::new();
    digest.update(source);
    digest.update([0]);
    digest.update(declaration.key.name.as_bytes());
    digest.update([0]);
    digest.update(kind.as_bytes());
    digest.update([0]);
    digest.update(name.as_bytes());
    Some(format!("sha256:{}", hex::encode(digest.finalize())))
}
