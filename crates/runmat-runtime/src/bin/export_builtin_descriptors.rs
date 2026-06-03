use std::env;
use std::fs;
use std::path::PathBuf;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinErrorDescriptor, BuiltinOutputMode, BuiltinSignatureDescriptor,
};
use serde::Serialize;

#[derive(Serialize)]
struct BuiltinDescriptorExport<'a> {
    name: &'static str,
    output_mode: BuiltinOutputMode,
    completion_policy: BuiltinCompletionPolicy,
    signatures: &'a [BuiltinSignatureDescriptor],
    errors: &'a [BuiltinErrorDescriptor],
}

#[derive(Serialize)]
struct BuiltinDescriptorFile<'a> {
    version: u8,
    builtins: Vec<BuiltinDescriptorExport<'a>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output_path: Option<PathBuf> = None;
    let mut include_non_public = false;

    for arg in env::args_os().skip(1) {
        if arg == "--include-non-public" {
            include_non_public = true;
        } else if output_path.is_none() {
            output_path = Some(PathBuf::from(arg));
        } else {
            return Err(format!("unexpected argument: {}", PathBuf::from(arg).display()).into());
        }
    }

    // Referencing the runtime crate ensures the inventory-backed builtin modules
    // are linked before we read runmat-builtins' global registry.
    let _ = runmat_runtime::object_property_getter_name("__descriptor_export_probe");

    let mut builtins = runmat_builtins::builtin_functions()
        .into_iter()
        .filter_map(|builtin| {
            let descriptor = builtin.descriptor?;
            if !include_non_public
                && descriptor.completion_policy != BuiltinCompletionPolicy::Public
            {
                return None;
            }
            Some(BuiltinDescriptorExport {
                name: builtin.name,
                output_mode: descriptor.output_mode,
                completion_policy: descriptor.completion_policy,
                signatures: descriptor.signatures,
                errors: descriptor.errors,
            })
        })
        .collect::<Vec<_>>();

    builtins.sort_by(|a, b| a.name.cmp(b.name));

    let payload = BuiltinDescriptorFile {
        version: 1,
        builtins,
    };
    let json = serde_json::to_string_pretty(&payload)?;

    if let Some(path) = output_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, json)?;
    } else {
        println!("{json}");
    }

    Ok(())
}
