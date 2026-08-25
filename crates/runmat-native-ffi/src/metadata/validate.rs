use std::collections::BTreeSet;

use thiserror::Error;

use crate::model::{NativeType, ParameterDirection, PointerOwnership};

use super::{NativeLibraryMetadata, NATIVE_FFI_METADATA_SCHEMA_VERSION};

#[derive(Debug, Error)]
pub enum MetadataError {
    #[error("invalid native-library metadata at {field}: {message}")]
    Invalid { field: String, message: String },
    #[error("could not serialize native-library metadata: {0}")]
    Serialize(serde_json::Error),
}

pub fn validate_metadata(metadata: &NativeLibraryMetadata) -> Result<(), MetadataError> {
    if metadata.schema_version != NATIVE_FFI_METADATA_SCHEMA_VERSION {
        return invalid(
            "schema_version",
            format!(
                "unsupported version {}; expected {}",
                metadata.schema_version, NATIVE_FFI_METADATA_SCHEMA_VERSION
            ),
        );
    }
    token("target_triple", &metadata.target_triple, 128)?;
    digest("source_digest", &metadata.source_digest)?;
    sorted_unique(
        "libraries",
        metadata
            .libraries
            .iter()
            .map(|library| library.name.as_str()),
    )?;
    sorted_unique(
        "structures",
        metadata
            .structures
            .iter()
            .map(|record| record.name.as_str()),
    )?;
    sorted_unique(
        "enumerations",
        metadata
            .enumerations
            .iter()
            .map(|enumeration| enumeration.name.as_str()),
    )?;
    sorted_unique(
        "aliases",
        metadata.aliases.iter().map(|alias| alias.name.as_str()),
    )?;

    let structures = metadata
        .structures
        .iter()
        .map(|record| record.name.as_str())
        .collect::<BTreeSet<_>>();
    let enumerations = metadata
        .enumerations
        .iter()
        .map(|enumeration| enumeration.name.as_str())
        .collect::<BTreeSet<_>>();

    for (library_index, library) in metadata.libraries.iter().enumerate() {
        let path = format!("libraries[{library_index}]");
        token(&format!("{path}.name"), &library.name, 128)?;
        nonempty(&format!("{path}.path"), &library.path, 4096)?;
        sorted_unique(
            &format!("{path}.dependencies"),
            library.dependencies.iter().map(String::as_str),
        )?;
        sorted_unique(
            &format!("{path}.symbols"),
            library.symbols.iter().map(|symbol| symbol.name.as_str()),
        )?;
        for (symbol_index, symbol) in library.symbols.iter().enumerate() {
            let symbol_path = format!("{path}.symbols[{symbol_index}]");
            token(&format!("{symbol_path}.name"), &symbol.name, 256)?;
            token(
                &format!("{symbol_path}.exported_name"),
                &symbol.exported_name,
                1024,
            )?;
            validate_type(
                &format!("{symbol_path}.return_type"),
                &symbol.return_type,
                &structures,
                &enumerations,
                0,
            )?;
            match &symbol.return_type {
                NativeType::Pointer { .. } => {
                    if symbol.return_ownership.is_none() {
                        return invalid(
                            format!("{symbol_path}.return_ownership"),
                            "pointer returns require an explicit ownership contract",
                        );
                    }
                }
                _ if symbol.return_ownership.is_some() || symbol.return_nullable => {
                    return invalid(
                        format!("{symbol_path}.return_type"),
                        "return ownership and nullability apply only to pointer returns",
                    );
                }
                _ => {}
            }
            let mut parameters = BTreeSet::new();
            for (parameter_index, parameter) in symbol.parameters.iter().enumerate() {
                let parameter_path = format!("{symbol_path}.parameters[{parameter_index}]");
                token(&format!("{parameter_path}.name"), &parameter.name, 256)?;
                if !parameters.insert(parameter.name.as_str()) {
                    return invalid(
                        format!("{symbol_path}.parameters"),
                        format!("duplicate parameter `{}`", parameter.name),
                    );
                }
                validate_type(
                    &format!("{parameter_path}.type"),
                    &parameter.ty,
                    &structures,
                    &enumerations,
                    0,
                )?;
                if !matches!(parameter.direction, ParameterDirection::Input)
                    && !matches!(parameter.ty, NativeType::Pointer { .. })
                {
                    return invalid(
                        format!("{parameter_path}.direction"),
                        "output parameters must have pointer type",
                    );
                }
                if !matches!(parameter.ty, NativeType::Pointer { .. })
                    && !matches!(parameter.ownership, PointerOwnership::Borrowed)
                {
                    return invalid(
                        format!("{parameter_path}.ownership"),
                        "ownership applies only to pointer parameters",
                    );
                }
            }
        }
    }
    for (index, record) in metadata.structures.iter().enumerate() {
        let path = format!("structures[{index}]");
        token(&format!("{path}.name"), &record.name, 256)?;
        unique(
            &format!("{path}.fields"),
            record.fields.iter().map(|field| field.name.as_str()),
        )?;
        for (field_index, field) in record.fields.iter().enumerate() {
            token(
                &format!("{path}.fields[{field_index}].name"),
                &field.name,
                256,
            )?;
            validate_type(
                &format!("{path}.fields[{field_index}].type"),
                &field.ty,
                &structures,
                &enumerations,
                0,
            )?;
        }
    }
    for (index, enumeration) in metadata.enumerations.iter().enumerate() {
        let path = format!("enumerations[{index}]");
        token(&format!("{path}.name"), &enumeration.name, 256)?;
        unique(
            &format!("{path}.variants"),
            enumeration
                .variants
                .iter()
                .map(|variant| variant.name.as_str()),
        )?;
    }
    for (index, alias) in metadata.aliases.iter().enumerate() {
        let path = format!("aliases[{index}]");
        token(&format!("{path}.name"), &alias.name, 256)?;
        validate_type(
            &format!("{path}.target"),
            &alias.target,
            &structures,
            &enumerations,
            0,
        )?;
    }
    Ok(())
}

fn validate_type(
    field: &str,
    ty: &NativeType,
    structures: &BTreeSet<&str>,
    enumerations: &BTreeSet<&str>,
    depth: usize,
) -> Result<(), MetadataError> {
    if depth > 32 {
        return invalid(field, "type nesting exceeds 32 levels");
    }
    match ty {
        NativeType::Void | NativeType::Scalar { .. } => Ok(()),
        NativeType::Pointer { pointee, .. } => {
            validate_type(field, pointee, structures, enumerations, depth + 1)
        }
        NativeType::Array { element, length } => {
            if *length == 0 {
                return invalid(field, "array length must be non-zero");
            }
            validate_type(field, element, structures, enumerations, depth + 1)
        }
        NativeType::Structure { name } => {
            if structures.contains(name.as_str()) {
                Ok(())
            } else {
                invalid(field, format!("unknown structure `{name}`"))
            }
        }
        NativeType::Enumeration { name, storage } => {
            let _ = storage;
            if enumerations.contains(name.as_str()) {
                Ok(())
            } else {
                invalid(field, format!("unknown enumeration `{name}`"))
            }
        }
        NativeType::Callback {
            return_type,
            parameters,
            ..
        } => {
            validate_type(field, return_type, structures, enumerations, depth + 1)?;
            for parameter in parameters {
                validate_type(field, &parameter.ty, structures, enumerations, depth + 1)?;
            }
            Ok(())
        }
    }
}

fn sorted_unique<'a>(
    field: &str,
    values: impl IntoIterator<Item = &'a str>,
) -> Result<(), MetadataError> {
    let mut previous: Option<&str> = None;
    for value in values {
        token(field, value, 256)?;
        if previous.is_some_and(|previous| previous >= value) {
            return invalid(field, "entries must be sorted and unique");
        }
        previous = Some(value);
    }
    Ok(())
}

fn unique<'a>(field: &str, values: impl IntoIterator<Item = &'a str>) -> Result<(), MetadataError> {
    let mut seen = BTreeSet::new();
    for value in values {
        token(field, value, 256)?;
        if !seen.insert(value) {
            return invalid(field, "entries must be unique");
        }
    }
    Ok(())
}

fn token(field: &str, value: &str, limit: usize) -> Result<(), MetadataError> {
    nonempty(field, value, limit)?;
    if !value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':' | b'@')
    }) {
        return invalid(field, "contains unsupported characters");
    }
    Ok(())
}

fn nonempty(field: &str, value: &str, limit: usize) -> Result<(), MetadataError> {
    if value.is_empty() || value.len() > limit || value.contains('\0') {
        return invalid(field, format!("must contain 1 to {limit} non-NUL bytes"));
    }
    Ok(())
}

fn digest(field: &str, value: &str) -> Result<(), MetadataError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return invalid(field, "must be a 64-character hexadecimal SHA-256 digest");
    }
    Ok(())
}

fn invalid<T>(field: impl Into<String>, message: impl Into<String>) -> Result<T, MetadataError> {
    Err(MetadataError::Invalid {
        field: field.into(),
        message: message.into(),
    })
}
