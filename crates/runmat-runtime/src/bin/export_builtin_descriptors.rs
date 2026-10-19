use std::env;
use std::fs;
use std::path::PathBuf;

use runmat_builtins::{
    BuiltinCatalogAudience, BuiltinCatalogEntry, BuiltinCompletionPolicy, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinIntegerAuditDescriptor, BuiltinIntegerCapabilityDescriptor,
    BuiltinOutputMode, BuiltinSignatureDescriptor,
};
use serde::Serialize;

#[derive(Serialize)]
struct BuiltinDescriptorExport<'a> {
    name: &'static str,
    output_mode: BuiltinOutputMode,
    completion_policy: BuiltinCompletionPolicy,
    signatures: &'a [BuiltinSignatureDescriptor],
    errors: &'a [BuiltinErrorDescriptor],
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    extensions: &'a [BuiltinExtensionDescriptor],
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    integer_capabilities: &'a [BuiltinIntegerCapabilityDescriptor],
    #[serde(skip_serializing_if = "Option::is_none")]
    integer_audit: Option<&'a BuiltinIntegerAuditDescriptor>,
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

    let mut builtins = runmat_builtins::builtin_catalog_entries()
        .iter()
        .filter_map(|entry| {
            let descriptor = entry.descriptor;
            if !exports_entry(entry, include_non_public) {
                return None;
            }
            Some(BuiltinDescriptorExport {
                name: entry.identity.name,
                output_mode: descriptor.output_mode,
                completion_policy: descriptor.completion_policy,
                signatures: descriptor.signatures,
                errors: descriptor.errors,
                extensions: entry.extensions,
                integer_capabilities: entry.integer_capabilities,
                integer_audit: entry.integer_audit,
            })
        })
        .chain(
            runmat_builtins::builtin_functions()
                .into_iter()
                .filter_map(|builtin| {
                    let descriptor = builtin.descriptor?;
                    if !exports_policy(descriptor.completion_policy, include_non_public) {
                        return None;
                    }
                    Some(BuiltinDescriptorExport {
                        name: builtin.name,
                        output_mode: descriptor.output_mode,
                        completion_policy: descriptor.completion_policy,
                        signatures: descriptor.signatures,
                        errors: descriptor.errors,
                        extensions: builtin.extensions,
                        integer_capabilities: builtin.integer_capabilities,
                        integer_audit: builtin.integer_audit,
                    })
                }),
        )
        .collect::<Vec<_>>();

    builtins.sort_by(|a, b| a.name.cmp(b.name));
    validate_unique_names(builtins.iter().map(|builtin| builtin.name))?;

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

fn exports_policy(policy: BuiltinCompletionPolicy, include_non_public: bool) -> bool {
    include_non_public || policy == BuiltinCompletionPolicy::Public
}

fn exports_entry(entry: &BuiltinCatalogEntry, include_non_public: bool) -> bool {
    entry.audience == BuiltinCatalogAudience::Public
        && exports_policy(entry.descriptor.completion_policy, include_non_public)
}

fn validate_unique_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(format!("duplicate builtin descriptor registration: {name}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{exports_entry, exports_policy, validate_unique_names, BuiltinCompletionPolicy};

    #[test]
    fn default_export_is_public_only_and_explicit_export_includes_all_policies() {
        for (policy, publicly_exported) in [
            (BuiltinCompletionPolicy::Public, true),
            (BuiltinCompletionPolicy::MethodOnly, false),
            (BuiltinCompletionPolicy::HiddenInternal, false),
        ] {
            assert_eq!(exports_policy(policy, false), publicly_exported);
            assert!(exports_policy(policy, true));
        }
    }

    #[test]
    fn fixture_descriptors_remain_private_even_with_non_public_export() {
        let fixture = runmat_builtins::builtin_catalog_entry_by_name("OverIdx.subsref")
            .expect("canonical fixture descriptor");
        assert!(!exports_entry(fixture, false));
        assert!(!exports_entry(fixture, true));
    }

    #[test]
    fn duplicate_registrations_cannot_produce_an_ambiguous_catalog() {
        assert!(validate_unique_names(["dir", "fileparts"]).is_ok());
        assert_eq!(
            validate_unique_names(["dir", "fileparts", "dir"]),
            Err("duplicate builtin descriptor registration: dir".into())
        );
        assert!(validate_unique_names([]).is_ok());
    }
}
