use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::cli::{NativeInterfaceCommand, NativeInterfacePrepareArgs};
use crate::presentation;

pub fn execute(command: NativeInterfaceCommand) -> Result<()> {
    match command {
        NativeInterfaceCommand::Prepare(args) => prepare(args),
    }
}

fn prepare(args: NativeInterfacePrepareArgs) -> Result<()> {
    let request = runmat_native_ffi::NativeInterfacePreparation {
        interface_name: args.interface_name,
        library_name: args.library_name,
        library_path: args.library,
        primary_header: args.header,
        additional_headers: args.additional_headers,
        include_directories: args.include_directories,
        definitions: args.definitions,
        compiler_frontend: args.frontend,
    };
    let prepared = runmat_native_ffi::prepare_native_interface(&request)
        .context("could not prepare native interface")?;
    let output = args.output.unwrap_or_else(|| {
        runmat_native_ffi::NativeInterfaceArtifactManifest::path_for_library(&request.library_path)
    });
    if let Some(parent) = output.parent().filter(|parent| *parent != Path::new("")) {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "could not create native-interface output directory {}",
                parent.display()
            )
        })?;
    }
    prepared
        .manifest
        .publish(&output)
        .with_context(|| format!("could not publish native interface {}", output.display()))?;

    if args.json {
        let result = NativeInterfacePreparationOutput {
            interface_name: &prepared.manifest.interface_name,
            library_name: &prepared.manifest.metadata.libraries[0].name,
            library: &request.library_path,
            manifest: &output,
            target_triple: &prepared.manifest.target_triple,
            identity: prepared.manifest.identity.as_str(),
            library_digest: &prepared.manifest.library_digest,
            library_bytes: prepared.manifest.library_bytes,
            warnings: &prepared.warnings,
        };
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        if !prepared.warnings.is_empty() {
            eprintln!(
                "{} {}",
                presentation::stderr().warning("Compiler warnings:"),
                prepared.warnings
            );
        }
        println!(
            "{} {}",
            presentation::stdout().success("Prepared"),
            presentation::stdout().path(output.display())
        );
    }
    Ok(())
}

#[derive(Serialize)]
struct NativeInterfacePreparationOutput<'a> {
    interface_name: &'a str,
    library_name: &'a str,
    library: &'a Path,
    manifest: &'a Path,
    target_triple: &'a str,
    identity: &'a str,
    library_digest: &'a str,
    library_bytes: u64,
    warnings: &'a str,
}
