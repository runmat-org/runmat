//! `assembly.component-configuration`: the selected components use another configuration of
//! their file. `default` (or an empty value) goes back to the file's default configuration; any
//! other name must be one of the file's configurations (table rows as `<table>.<row>`), and
//! the line written is `configuration = { select = named, name = <name> }` in the component's block.

use super::{find, model};
use crate::runtime::{resolve_path, Files};
use runmat_cad::assembly::model::{AssemblyDocument, ConfigurationSelection, DefinitionKind};
use runmat_cad::ids::ConfigurationName;
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::syntax::parse_part;

/// The configuration names a part file offers, `default` first; None when the file isn't read.
fn offered(document: &AssemblyDocument, file: &str, files: &Files) -> Option<Vec<String>> {
    let path = resolve_path(&document.path.0, file, files);
    let text = files.get(&path)?;
    let part = parse_part(&path, text).document?;
    let mut names = vec!["default".to_owned()];
    names.extend(part.all_configurations().into_iter().map(|c| c.name.0));
    Some(names)
}

pub(super) fn configure(
    document: &mut AssemblyDocument,
    files: &Files,
    names: &[String],
    configuration: &str,
) -> Result<String, EditFailure> {
    let wanted = configuration.trim();
    for name in names {
        let at = find(document, name)?;
        let definition = &document.occurrences[at].source.definition;
        if definition.kind == DefinitionKind::Part && !wanted.is_empty() {
            if let Some(names) = offered(document, &definition.file.0, files) {
                if !names.iter().any(|n| n == wanted) {
                    return Err(model(format!(
                        "{} has no configuration `{wanted}`; it has {}",
                        definition.file.0.rsplit('/').next().unwrap_or(&definition.file.0),
                        names.join(", ")
                    )));
                }
            }
        }
        document.occurrences[at].configuration = match wanted {
            "" | "default" => ConfigurationSelection::Default,
            name => ConfigurationSelection::Named { name: ConfigurationName::new(name) },
        };
    }
    let shown = if wanted.is_empty() { "default" } else { wanted };
    Ok(format!("{} use{} {shown}", names.join(", "), if names.len() == 1 { "s" } else { "" }))
}
