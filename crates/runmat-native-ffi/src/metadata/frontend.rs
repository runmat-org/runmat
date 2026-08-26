use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::model::{
    CallingConvention, EnumerationDefinition, EnumerationVariant, NativeLibrary, NativeScalar,
    NativeType, Parameter, ParameterDirection, PointerMutability, PointerOwnership,
    StructureDefinition, StructureField, SymbolPrototype, TypeAliasDefinition,
};

use super::{
    normalize_metadata, validate_metadata, MetadataError, NativeLibraryMetadata,
    NATIVE_FFI_METADATA_SCHEMA_VERSION,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderPreparation {
    pub header: PathBuf,
    pub library_name: String,
    pub library_path: String,
    pub target_triple: String,
    pub clang: PathBuf,
    pub include_directories: Vec<PathBuf>,
    pub definitions: Vec<String>,
}

#[derive(Debug, Error)]
pub enum HeaderPreparationError {
    #[error("could not read native header {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not start compiler frontend {program}: {source}")]
    Start {
        program: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("compiler frontend rejected {header}: {diagnostic}")]
    Frontend { header: PathBuf, diagnostic: String },
    #[error("compiler frontend returned malformed syntax metadata: {0}")]
    Decode(serde_json::Error),
    #[error("unsupported declaration `{declaration}`: {message}")]
    Unsupported {
        declaration: String,
        message: String,
    },
    #[error(transparent)]
    Metadata(#[from] MetadataError),
}

pub fn prepare_header(
    preparation: &HeaderPreparation,
) -> Result<NativeLibraryMetadata, HeaderPreparationError> {
    let bytes = fs::read(&preparation.header).map_err(|source| HeaderPreparationError::Read {
        path: preparation.header.clone(),
        source,
    })?;
    let mut command = Command::new(&preparation.clang);
    command
        .arg("-x")
        .arg("c")
        .arg("-std=c11")
        .arg("-fsyntax-only")
        .arg("-Xclang")
        .arg("-ast-dump=json")
        .arg(format!("--target={}", preparation.target_triple));
    for directory in &preparation.include_directories {
        command.arg("-I").arg(directory);
    }
    for definition in &preparation.definitions {
        command.arg(format!("-D{definition}"));
    }
    command.arg(&preparation.header);
    let output = command
        .output()
        .map_err(|source| HeaderPreparationError::Start {
            program: preparation.clang.clone(),
            source,
        })?;
    if !output.status.success() {
        return Err(HeaderPreparationError::Frontend {
            header: preparation.header.clone(),
            diagnostic: bounded_diagnostic(&output.stderr),
        });
    }
    let ast: Value =
        serde_json::from_slice(&output.stdout).map_err(HeaderPreparationError::Decode)?;
    let canonical_header = preparation
        .header
        .canonicalize()
        .unwrap_or_else(|_| preparation.header.clone());
    let mut declarations = Declarations::default();
    declarations.visit(&ast, &canonical_header)?;
    let metadata = normalize_metadata(NativeLibraryMetadata {
        schema_version: NATIVE_FFI_METADATA_SCHEMA_VERSION,
        target_triple: preparation.target_triple.clone(),
        source_digest: format!("{:x}", Sha256::digest(bytes)),
        libraries: vec![NativeLibrary {
            name: preparation.library_name.clone(),
            path: preparation.library_path.clone(),
            dependencies: Vec::new(),
            symbols: declarations.functions,
        }],
        structures: declarations.structures,
        enumerations: declarations.enumerations,
        aliases: declarations.aliases,
    });
    validate_metadata(&metadata)?;
    Ok(metadata)
}

#[derive(Default)]
struct Declarations {
    functions: Vec<SymbolPrototype>,
    structures: Vec<StructureDefinition>,
    enumerations: Vec<EnumerationDefinition>,
    aliases: Vec<TypeAliasDefinition>,
    seen: BTreeSet<(String, String)>,
    typedefs: BTreeMap<String, String>,
}

impl Declarations {
    fn visit(&mut self, node: &Value, header: &Path) -> Result<(), HeaderPreparationError> {
        if originates_in(node, header) {
            match string(node, "kind") {
                Some("TypedefDecl") => self.typedef(node)?,
                Some("RecordDecl") => self.record(node)?,
                Some("EnumDecl") => self.enumeration(node)?,
                Some("FunctionDecl") => self.function(node)?,
                _ => {}
            }
        }
        if let Some(children) = node.get("inner").and_then(Value::as_array) {
            for child in children {
                self.visit(child, header)?;
            }
        }
        Ok(())
    }

    fn typedef(&mut self, node: &Value) -> Result<(), HeaderPreparationError> {
        let Some(name) = string(node, "name") else {
            return Ok(());
        };
        let Some(qual_type) = type_spelling(node) else {
            return Ok(());
        };
        self.typedefs.insert(name.into(), qual_type.into());
        if is_builtin_typedef(name) {
            return Ok(());
        }
        let target = parse_type(qual_type, &self.typedefs)?;
        if self.seen.insert(("alias".into(), name.into())) {
            self.aliases.push(TypeAliasDefinition {
                name: name.into(),
                target,
            });
        }
        Ok(())
    }

    fn record(&mut self, node: &Value) -> Result<(), HeaderPreparationError> {
        if !node
            .get("completeDefinition")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Ok(());
        }
        let Some(name) = string(node, "name") else {
            return Ok(());
        };
        if !self.seen.insert(("record".into(), name.into())) {
            return Ok(());
        }
        let mut fields = Vec::new();
        for child in children(node) {
            if string(child, "kind") != Some("FieldDecl") {
                continue;
            }
            let field_name = required_string(child, "name", name)?;
            let field_type = required_type(child, field_name, &self.typedefs)?;
            fields.push(StructureField {
                name: field_name.into(),
                ty: field_type,
            });
        }
        self.structures.push(StructureDefinition {
            name: name.into(),
            fields,
        });
        Ok(())
    }

    fn enumeration(&mut self, node: &Value) -> Result<(), HeaderPreparationError> {
        let Some(name) = string(node, "name") else {
            return Ok(());
        };
        if !self.seen.insert(("enum".into(), name.into())) {
            return Ok(());
        }
        let mut variants = Vec::new();
        let mut next_value = 0_i64;
        for child in children(node) {
            if string(child, "kind") != Some("EnumConstantDecl") {
                continue;
            }
            let variant_name = required_string(child, "name", name)?;
            let value = enum_value(child).unwrap_or(next_value);
            variants.push(EnumerationVariant {
                name: variant_name.into(),
                value,
            });
            next_value = value.saturating_add(1);
        }
        self.enumerations.push(EnumerationDefinition {
            name: name.into(),
            storage: NativeScalar::I32,
            variants,
        });
        Ok(())
    }

    fn function(&mut self, node: &Value) -> Result<(), HeaderPreparationError> {
        let Some(name) = string(node, "name") else {
            return Ok(());
        };
        if !self.seen.insert(("function".into(), name.into())) {
            return Ok(());
        }
        let signature = type_spelling(node).ok_or_else(|| HeaderPreparationError::Unsupported {
            declaration: name.into(),
            message: "missing function type".into(),
        })?;
        let return_spelling = signature
            .split_once('(')
            .map(|(return_type, _)| return_type)
            .ok_or_else(|| HeaderPreparationError::Unsupported {
                declaration: name.into(),
                message: format!("could not separate return type from `{signature}`"),
            })?;
        let mut parameters = Vec::new();
        for (index, child) in children(node).iter().enumerate() {
            if string(child, "kind") != Some("ParmVarDecl") {
                continue;
            }
            let parameter_name = string(child, "name")
                .map(str::to_owned)
                .unwrap_or_else(|| format!("arg{index}"));
            let ty = required_type(child, &parameter_name, &self.typedefs)?;
            let direction = match &ty {
                NativeType::Pointer { mutability, .. } => match mutability {
                    PointerMutability::Const => ParameterDirection::Input,
                    PointerMutability::Mutable => ParameterDirection::InputOutput,
                },
                _ => ParameterDirection::Input,
            };
            parameters.push(Parameter {
                name: parameter_name,
                ty,
                direction,
                ownership: PointerOwnership::Borrowed,
                nullable: false,
            });
        }
        let return_type = parse_type(return_spelling, &self.typedefs)?;
        // A C declaration describes the pointee type, but it does not prove
        // who owns the returned allocation or whether a null address is
        // possible. Prepared declarations therefore use the only contract
        // that can be inferred safely: a nullable borrowed reference.
        let returns_pointer = matches!(return_type, NativeType::Pointer { .. });
        let return_ownership = returns_pointer.then_some(PointerOwnership::Borrowed);
        self.functions.push(SymbolPrototype {
            name: name.into(),
            exported_name: name.into(),
            calling_convention: calling_convention(signature),
            return_type,
            return_ownership,
            return_nullable: returns_pointer,
            parameters,
            variadic: signature.contains(", ...)") || signature.ends_with("(...)"),
        });
        Ok(())
    }
}

fn parse_type(
    spelling: &str,
    typedefs: &BTreeMap<String, String>,
) -> Result<NativeType, HeaderPreparationError> {
    let spelling = spelling.trim();
    if let Some(target) = typedefs.get(spelling) {
        if target != spelling {
            return parse_type(target, typedefs);
        }
    }
    if let Some((element, length)) = parse_array(spelling) {
        return Ok(NativeType::Array {
            element: Box::new(parse_type(element, typedefs)?),
            length,
        });
    }
    if let Some(callback) = parse_callback(spelling, typedefs)? {
        return Ok(callback);
    }
    if let Some(pointee) = spelling.strip_suffix('*') {
        let pointee = pointee.trim();
        let mutability = if pointee.starts_with("const ") || pointee.ends_with(" const") {
            PointerMutability::Const
        } else {
            PointerMutability::Mutable
        };
        let pointee = if let Some(unqualified) = pointee.strip_prefix("const ") {
            unqualified
        } else if let Some(unqualified) = pointee.strip_suffix(" const") {
            unqualified
        } else {
            pointee
        }
        .trim();
        return Ok(NativeType::Pointer {
            pointee: Box::new(parse_type(pointee, typedefs)?),
            mutability,
        });
    }
    let normalized = spelling
        .trim_start_matches("const ")
        .trim_end_matches(" const")
        .trim();
    let scalar = match normalized {
        "void" => return Ok(NativeType::Void),
        "_Bool" | "bool" => NativeScalar::Bool,
        "char" => NativeScalar::Char,
        "signed char" => NativeScalar::SignedChar,
        "unsigned char" => NativeScalar::UnsignedChar,
        "short" | "short int" | "signed short" => NativeScalar::Short,
        "unsigned short" | "unsigned short int" => NativeScalar::UnsignedShort,
        "int" | "signed" | "signed int" => NativeScalar::Int,
        "unsigned" | "unsigned int" => NativeScalar::UnsignedInt,
        "long" | "long int" | "signed long" => NativeScalar::Long,
        "unsigned long" | "unsigned long int" => NativeScalar::UnsignedLong,
        "long long" | "long long int" | "signed long long" => NativeScalar::LongLong,
        "unsigned long long" | "unsigned long long int" => NativeScalar::UnsignedLongLong,
        "int8_t" => NativeScalar::I8,
        "uint8_t" => NativeScalar::U8,
        "int16_t" => NativeScalar::I16,
        "uint16_t" => NativeScalar::U16,
        "int32_t" => NativeScalar::I32,
        "uint32_t" => NativeScalar::U32,
        "int64_t" => NativeScalar::I64,
        "uint64_t" => NativeScalar::U64,
        "intptr_t" | "ptrdiff_t" | "ssize_t" => NativeScalar::Isize,
        "uintptr_t" | "size_t" => NativeScalar::Usize,
        "float" => NativeScalar::F32,
        "double" => NativeScalar::F64,
        _ if normalized.starts_with("struct ") => {
            return Ok(NativeType::Structure {
                name: normalized[7..].trim().into(),
            })
        }
        _ if normalized.starts_with("enum ") => {
            return Ok(NativeType::Enumeration {
                name: normalized[5..].trim().into(),
                storage: NativeScalar::I32,
            })
        }
        _ => {
            return Err(HeaderPreparationError::Unsupported {
                declaration: normalized.into(),
                message: "type is outside the supported C ABI contract".into(),
            })
        }
    };
    Ok(NativeType::Scalar { scalar })
}

fn parse_callback(
    spelling: &str,
    typedefs: &BTreeMap<String, String>,
) -> Result<Option<NativeType>, HeaderPreparationError> {
    let Some(pointer_open) = spelling.find("(*") else {
        return Ok(None);
    };
    let Some(parameters_open_relative) = spelling[pointer_open..].find(")(") else {
        return Ok(None);
    };
    let parameters_open = pointer_open + parameters_open_relative + 2;
    let Some(parameters_end) = spelling.rfind(')') else {
        return Ok(None);
    };
    if parameters_end < parameters_open {
        return Ok(None);
    }
    let return_type = parse_type(spelling[..pointer_open].trim(), typedefs)?;
    let parameter_spelling = spelling[parameters_open..parameters_end].trim();
    let parameters = if parameter_spelling.is_empty() || parameter_spelling == "void" {
        Vec::new()
    } else {
        parameter_spelling
            .split(',')
            .enumerate()
            .map(|(index, spelling)| {
                Ok(Parameter {
                    name: format!("arg{index}"),
                    ty: parse_type(spelling.trim(), typedefs)?,
                    direction: ParameterDirection::Input,
                    ownership: PointerOwnership::Borrowed,
                    nullable: false,
                })
            })
            .collect::<Result<Vec<_>, HeaderPreparationError>>()?
    };
    Ok(Some(NativeType::Callback {
        calling_convention: calling_convention(spelling),
        return_type: Box::new(return_type),
        parameters,
    }))
}

fn parse_array(spelling: &str) -> Option<(&str, usize)> {
    let open = spelling.rfind('[')?;
    let length = spelling.get(open + 1..spelling.len().checked_sub(1)?)?;
    if !spelling.ends_with(']') {
        return None;
    }
    Some((spelling[..open].trim(), length.trim().parse().ok()?))
}

fn calling_convention(signature: &str) -> CallingConvention {
    if signature.contains("__attribute__((stdcall))") {
        CallingConvention::Stdcall
    } else if signature.contains("__attribute__((fastcall))") {
        CallingConvention::Fastcall
    } else if signature.contains("__attribute__((thiscall))") {
        CallingConvention::Thiscall
    } else if signature.contains("__attribute__((vectorcall))") {
        CallingConvention::Vectorcall
    } else {
        CallingConvention::C
    }
}

fn originates_in(node: &Value, header: &Path) -> bool {
    let location = node.get("loc");
    let range_begin = node.pointer("/range/begin");
    let file = location
        .and_then(|value| value.get("file"))
        .and_then(Value::as_str)
        .or_else(|| {
            range_begin
                .and_then(|value| value.get("file"))
                .and_then(Value::as_str)
        });
    if let Some(file) = file {
        let candidate = Path::new(file)
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(file));
        return candidate == header;
    }
    let has_main_file_offset = location
        .and_then(|value| value.get("offset"))
        .and_then(Value::as_u64)
        .or_else(|| {
            range_begin
                .and_then(|value| value.get("offset"))
                .and_then(Value::as_u64)
        })
        .is_some();
    let included = location
        .and_then(|value| value.get("includedFrom"))
        .or_else(|| range_begin.and_then(|value| value.get("includedFrom")))
        .is_some();
    has_main_file_offset && !included
}

fn children(node: &Value) -> &[Value] {
    node.get("inner")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn string<'a>(node: &'a Value, field: &str) -> Option<&'a str> {
    node.get(field).and_then(Value::as_str)
}

fn type_spelling(node: &Value) -> Option<&str> {
    node.pointer("/type/desugaredQualType")
        .and_then(Value::as_str)
        .or_else(|| node.pointer("/type/qualType").and_then(Value::as_str))
}

fn required_string<'a>(
    node: &'a Value,
    field: &str,
    declaration: &str,
) -> Result<&'a str, HeaderPreparationError> {
    string(node, field).ok_or_else(|| HeaderPreparationError::Unsupported {
        declaration: declaration.into(),
        message: format!("missing {field}"),
    })
}

fn required_type(
    node: &Value,
    declaration: &str,
    typedefs: &BTreeMap<String, String>,
) -> Result<NativeType, HeaderPreparationError> {
    let spelling = type_spelling(node).ok_or_else(|| HeaderPreparationError::Unsupported {
        declaration: declaration.into(),
        message: "missing type".into(),
    })?;
    parse_type(spelling, typedefs)
}

fn enum_value(node: &Value) -> Option<i64> {
    descendants(node).find_map(|child| {
        child
            .get("value")
            .and_then(Value::as_str)
            .and_then(|value| value.parse().ok())
    })
}

fn descendants(node: &Value) -> impl Iterator<Item = &Value> {
    let mut pending = children(node).iter().rev().collect::<Vec<_>>();
    std::iter::from_fn(move || {
        let next = pending.pop()?;
        pending.extend(children(next).iter().rev());
        Some(next)
    })
}

fn is_builtin_typedef(name: &str) -> bool {
    name.starts_with("__") || matches!(name, "size_t" | "ptrdiff_t" | "intptr_t" | "uintptr_t")
}

fn bounded_diagnostic(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut result = text.chars().take(16_384).collect::<String>();
    if text.chars().count() > 16_384 {
        result.push_str("\n[diagnostic truncated]");
    }
    result
}
