use std::path::PathBuf;

use crate::BuiltinResult;

pub(super) struct Inventory {
    pub folder: PathBuf,
    pub sources: Vec<String>,
    pub data: Vec<String>,
    pub extensions: Vec<String>,
    pub classes: Vec<String>,
    pub packages: Vec<String>,
}

pub(super) async fn inspect(folder: PathBuf) -> BuiltinResult<Inventory> {
    let entries = runmat_filesystem::read_dir_async(&folder)
        .await
        .map_err(|error| {
            super::error::message(
                &runmat_builtins::WHAT_ERROR_FILESYSTEM,
                format!("what: unable to inspect '{}' ({error})", folder.display()),
            )
        })?;
    let mut inventory = Inventory {
        folder,
        sources: Vec::new(),
        data: Vec::new(),
        extensions: Vec::new(),
        classes: Vec::new(),
        packages: Vec::new(),
    };
    for entry in entries {
        classify(
            &mut inventory,
            &entry.file_name().to_string_lossy(),
            entry.is_dir(),
        );
    }
    inventory.sources.sort();
    inventory.data.sort();
    inventory.extensions.sort();
    inventory.classes.sort();
    inventory.packages.sort();
    Ok(inventory)
}

fn classify(inventory: &mut Inventory, name: &str, directory: bool) {
    if directory {
        if let Some(class) = name.strip_prefix('@').filter(|name| !name.is_empty()) {
            inventory.classes.push(class.to_owned());
        } else if let Some(package) = name.strip_prefix('+').filter(|name| !name.is_empty()) {
            inventory.packages.push(package.to_owned());
        }
    } else if name.ends_with(".m") {
        inventory.sources.push(name.to_owned());
    } else if name.ends_with(".mat") {
        inventory.data.push(name.to_owned());
    } else if name.contains(".mex") {
        inventory.extensions.push(name.to_owned());
    }
}
