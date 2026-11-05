//! `design.export` and `design.import`.
//!
//! Exports: parts and assemblies as STEP (AP242, the exact B-rep the installed kernel regenerated;
//! a faceted AP214 B-rep in millimetres only on the grid test kernel), STL (binary, millimetres), 3MF (millimetres) and glTF (binary `.glb`, metres,
//! Y up as glTF defines); drawing sheets and sheet-metal flat patterns as DXF (millimetres);
//! circuits and boards as a KiCad project. Imports: KiCad projects to `.circuit`,
//! `.circuit.layout` and `.circuit.lock` sources; STEP files to a `.part` whose `import` feature
//! names the file (the names inside the STEP file become the part's face names).

use crate::edit::{stage, StagedChange};
use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use base64::Engine;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

mod dxf;
mod exact_step;
mod mesh;
mod step;

use dxf::{drawing_dxf, flat_pattern_dxf};
use mesh::floats;
pub use mesh::{glb, model_meshes, stl, three_mf, zip_stored, Mesh};
pub use step::step;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Step,
    Stl,
    #[serde(rename = "3mf")]
    ThreeMf,
    Gltf,
    Dxf,
    Kicad,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExportInput {
    /// Workspace path of the design file.
    pub path: String,
    pub format: ExportFormat,
    /// Drawings: the sheet to export (default the first).
    #[serde(default)]
    pub sheet: Option<String>,
    /// KiCad: the project name (default the circuit's file stem).
    #[serde(default)]
    pub project_name: Option<String>,
}

/// One exported file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExportedFile {
    pub name: String,
    pub mime_type: String,
    /// `utf8` or `base64`.
    pub encoding: String,
    pub content: String,
    pub bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExportOutput {
    pub path: String,
    pub format: ExportFormat,
    pub files: Vec<ExportedFile>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

fn stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    for ext in [
        ".circuit.layout",
        ".circuit",
        ".part",
        ".assembly",
        ".drawing",
    ] {
        if let Some(s) = name.strip_suffix(ext) {
            return s.to_owned();
        }
    }
    name.to_owned()
}

fn text_file(name: String, mime: &str, content: String) -> ExportedFile {
    ExportedFile {
        bytes: content.len(),
        name,
        mime_type: mime.into(),
        encoding: "utf8".into(),
        content,
    }
}

fn binary_file(name: String, mime: &str, bytes: &[u8]) -> ExportedFile {
    ExportedFile {
        bytes: bytes.len(),
        name,
        mime_type: mime.into(),
        encoding: "base64".into(),
        content: base64::engine::general_purpose::STANDARD.encode(bytes),
    }
}

pub fn export(input: &ExportInput, files: &Files) -> Result<ExportOutput, ToolError> {
    let kind = kind_of(&input.path)?;
    let text = text_of(files, &input.path)?;
    let name = stem(&input.path);
    let mut notes = Vec::new();
    let files_out = match input.format {
        ExportFormat::Step | ExportFormat::Stl | ExportFormat::ThreeMf | ExportFormat::Gltf => {
            let meshes = model_meshes(&input.path, files)?;
            match input.format {
                ExportFormat::Step => match exact_step::exact_step(&input.path, files)? {
                    Some(bytes) => {
                        notes.push("STEP AP242: the exact B-rep of the regenerated bodies".into());
                        vec![text_file(
                            format!("{name}.step"),
                            "model/step",
                            String::from_utf8_lossy(&bytes).into_owned(),
                        )]
                    }
                    None => {
                        notes.push(
                            "STEP AP214 faceted B-rep (the grid test kernel has no exact STEP writer), millimetres".into(),
                        );
                        vec![text_file(format!("{name}.step"), "model/step", step(&meshes, &name))]
                    }
                },
                ExportFormat::Stl => vec![binary_file(
                    format!("{name}.stl"),
                    "model/stl",
                    &stl(&meshes, &name),
                )],
                ExportFormat::ThreeMf => vec![binary_file(
                    format!("{name}.3mf"),
                    "model/3mf",
                    &three_mf(&meshes),
                )],
                _ => vec![binary_file(
                    format!("{name}.glb"),
                    "model/gltf-binary",
                    &glb(&meshes),
                )],
            }
        }
        ExportFormat::Dxf => match kind {
            DesignKind::Drawing => {
                let (sheet, dxf, n) =
                    drawing_dxf(&input.path, text, files, input.sheet.as_deref())?;
                notes.extend(n);
                vec![text_file(
                    format!("{name}-{sheet}.dxf"),
                    "image/vnd.dxf",
                    dxf,
                )]
            }
            DesignKind::Part => vec![text_file(
                format!("{name}-flat.dxf"),
                "image/vnd.dxf",
                flat_pattern_dxf(&input.path, text)?,
            )],
            other => {
                return Err(ToolError::new(
                    "unsupported",
                    format!(
                        "DXF exports drawings and part flat patterns; {} is a {other:?} file",
                        input.path
                    ),
                ))
            }
        },
        ExportFormat::Kicad => {
            if !kind.is_circuit() {
                return Err(ToolError::new(
                    "unsupported",
                    format!(
                        "KiCad exports circuits and boards; {} is a {kind:?} file",
                        input.path
                    ),
                ));
            }
            let project = input.project_name.clone().unwrap_or_else(|| name.clone());
            let out = runmat_circuit::runtime::export_kicad(
                &input.path,
                text,
                files,
                &Default::default(),
                &project,
            )
            .map_err(|d| {
                ToolError::new(
                    "not_compiled",
                    format!(
                        "{} doesn't compile (design.check lists why): {}",
                        input.path,
                        d.first().map(|d| d.message.as_str()).unwrap_or("")
                    ),
                )
            })?;
            out.into_iter()
                .map(|(n, t)| text_file(n, "application/x-kicad", t))
                .collect()
        }
    };
    Ok(ExportOutput {
        path: input.path.clone(),
        format: input.format,
        files: files_out,
        notes,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImportFormat {
    Kicad,
    Step,
}

/// A file handed to `design.import`.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImportFile {
    /// File name (`board.kicad_pcb`, `root.kicad_sch`, `housing.step`).
    pub name: String,
    pub content: String,
    /// `utf8` (default) or `base64`.
    #[serde(default)]
    pub encoding: Option<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImportInput {
    pub format: ImportFormat,
    pub files: Vec<ImportFile>,
    /// KiCad: the target stem (`boards/io` writes `boards/io.circuit`, `.circuit.layout` and
    /// `.circuit.lock`). STEP: the `.part` to write (`parts/housing.part`); the STEP file goes
    /// beside it.
    pub target: String,
    /// KiCad: the root schematic's file name (default: the schematic no other sheet names).
    #[serde(default)]
    pub root: Option<String>,
}

fn decode(file: &ImportFile) -> Result<String, ToolError> {
    match file.encoding.as_deref() {
        Some("base64") => {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(file.content.as_bytes())
                .map_err(|e| ToolError::invalid(format!("{}: not base64: {e}", file.name)))?;
            String::from_utf8(bytes).map_err(|_| {
                ToolError::invalid(format!(
                    "{} isn't text (STEP and KiCad files are)",
                    file.name
                ))
            })
        }
        _ => Ok(file.content.clone()),
    }
}

pub fn import(input: &ImportInput, files: &Files) -> Result<StagedChange, ToolError> {
    let target = input.target.trim_start_matches("./").to_owned();
    match input.format {
        ImportFormat::Kicad => {
            let mut kicad = BTreeMap::new();
            for f in &input.files {
                kicad.insert(f.name.clone(), decode(f)?);
            }
            let stem = target
                .strip_suffix(".circuit.layout")
                .or_else(|| target.strip_suffix(".circuit"))
                .unwrap_or(&target)
                .to_owned();
            let read = runmat_circuit::runtime::import_kicad(&kicad, input.root.as_deref(), &stem)
                .map_err(ToolError::invalid)?;
            let primary = read
                .files
                .keys()
                .find(|k| k.ends_with(".circuit"))
                .or_else(|| read.files.keys().next())
                .cloned()
                .unwrap_or_default();
            let summary = format!(
                "Import KiCad project as {}",
                read.files.keys().cloned().collect::<Vec<_>>().join(", ")
            );
            let mut change = stage(summary, read.files.into_iter().collect(), files, &primary);
            change.notes.extend(read.warnings);
            Ok(change)
        }
        ImportFormat::Step => {
            let Some(file) = input.files.first() else {
                return Err(ToolError::invalid("give the STEP file in `files`"));
            };
            let content = decode(file)?;
            if !content.trim_start().starts_with("ISO-10303-21") {
                return Err(ToolError::invalid(format!(
                    "{} isn't a STEP file (ISO-10303-21)",
                    file.name
                )));
            }
            let part_path = if target.ends_with(".part") {
                target.clone()
            } else {
                format!("{target}.part")
            };
            let folder = part_path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
            let step_name = file
                .name
                .rsplit('/')
                .next()
                .unwrap_or(&file.name)
                .to_owned();
            let step_path = if folder.is_empty() {
                step_name.clone()
            } else {
                format!("{folder}/{step_name}")
            };
            let digest = hex::encode(&Sha256::digest(content.as_bytes())[..8]);
            let part_name: String = stem(&part_path)
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() {
                        c.to_ascii_lowercase()
                    } else {
                        '_'
                    }
                })
                .collect();
            let part = format!(
                "runmat 0.7\nstd 1\n\npart {part_name} {{\n    feature step = import(source = \"{step_name}\", digest = sha256_{digest})\n}}\n"
            );
            let summary = format!("Import {step_name} as {part_path}");
            let mut change = stage(
                summary,
                vec![(step_path, content), (part_path.clone(), part)],
                files,
                &part_path,
            );
            change.notes.push(
                "the STEP file is the source of truth: its solid and face names become the part's (feature.step.face(\"<name>\"))"
                    .into(),
            );
            Ok(change)
        }
    }
}
