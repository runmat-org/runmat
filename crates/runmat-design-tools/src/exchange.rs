//! `design.export` and `design.import`.
//!
//! Exports: parts and assemblies as STEP (AP242, the exact B-rep the installed kernel regenerated,
//! with the part's semantic PMI;
//! a faceted AP214 B-rep in millimetres only on the grid test kernel), STL (binary, millimetres), 3MF (millimetres) and glTF (binary `.glb`, metres,
//! Y up as glTF defines); drawing sheets and sheet-metal flat patterns as DXF and DWG (R2018; millimetres), drawing sheets as vector PDF;
//! assemblies' bills of materials as CSV; circuits and boards as a KiCad project; boards as a
//! fabrication set ([`fabrication`]: Gerber X2, Excellon, drill map, fabrication drawing,
//! pick-and-place and BOM per variant, IPC-2581C, ODB++, IPC-D-356A); circuits as KiCad,
//! SPICE and EDIF 2.0.0 netlists ([`netlist`]). A product
//! family exports every configuration at once (`allConfigurations`). Imports: KiCad projects to `.circuit`,
//! `.circuit.layout` and `.circuit.lock` sources; STEP files to a `.part` whose `import` feature
//! names the file (the names inside the STEP file become the part's face names, and its AP242
//! semantic PMI becomes the part's datums, tolerances, sizes and locations on those faces).

use crate::edit::{stage, StagedChange};
use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use base64::Engine;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub mod dxf;
mod exact_step;
pub mod fabrication;
mod face_dxf;
pub mod family;
pub mod host;
pub mod netlist;
mod mesh;
pub mod schematic_pdf;
pub mod sheet;
mod step;
mod three_mf;

use dxf::{drawing_dxf, flat_pattern_dxf};
use mesh::floats;
pub use mesh::{glb, model_meshes, model_scene, model_scene_in, stl, Mesh, Placed, Scene3d, Shape};
pub use three_mf::{scene_model, three_mf};
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
    /// DWG (R2018 format) of a drawing sheet or a part's flat pattern.
    Dwg,
    /// A drawing sheet as a one-page vector PDF at true size, or a circuit's schematic as a
    /// PDF set: every page, with its title block, in one file.
    Pdf,
    /// An assembly's flat bill of materials as CSV.
    Bom,
    Kicad,
    /// A board's fabrication and assembly outputs, as one folder of files.
    Fabrication,
    /// A circuit's KiCad netlist (`.net`).
    KicadNetlist,
    /// A circuit's SPICE deck (`.cir`); parts without a model are left out and listed.
    Spice,
    /// A circuit's EDIF 2.0.0 netlist (`.edn`).
    Edif,
    /// A board's IPC-D-356A test netlist (`.ipc`).
    Ipc356,
}

/// A fabrication export's settings. Each field given overrides the board's saved settings
/// (its layout's `fabrication = { … }`), which start from the fab-house preset when the board
/// has none.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FabricationOptions {
    /// `fab_house`, `assembly`, `full` or `custom`: sets the outputs before the flags below.
    pub preset: Option<String>,
    /// Gerber X2 per layer and the job file with the stackup.
    pub gerber: Option<bool>,
    /// Excellon drill files, plated per span and non-plated.
    pub drill: Option<bool>,
    /// The drill map (Gerber and PDF).
    pub drill_map: Option<bool>,
    /// The fabrication drawing with the layer and stackup table (Gerber and PDF).
    pub drawing: Option<bool>,
    /// Pick-and-place (centroid) CSV per variant.
    pub pick_place: Option<bool>,
    /// Bill of materials CSV per variant.
    pub bom: Option<bool>,
    /// The assembly drawing of each side (Gerber and PDF).
    pub assembly_drawing: Option<bool>,
    /// IPC-2581C XML.
    pub ipc2581: Option<bool>,
    /// ODB++ job as a .tgz.
    pub odb: Option<bool>,
    /// IPC-D-356A netlist.
    pub ipc356: Option<bool>,
    /// Pick-and-place lists through-hole parts too.
    pub include_through_hole: Option<bool>,
    /// The variants to make assembly files for (`""` is the default); all when absent or empty.
    pub variants: Option<Vec<String>>,
    /// The Gerber layers to write, by name (`F.Cu`, `F.Courtyard`); the default layers when absent.
    pub layers: Option<Vec<String>>,
    /// The output folder, relative to the board's folder (`fab/<board>` when absent).
    pub folder: Option<String>,
    /// Also pack every file into `<board>-fab.zip`.
    pub zip: Option<bool>,
    /// The board revision: named in the Gerber and job files and the default folder
    /// (`fab/<board>-<revision>`); an empty text clears it.
    pub revision: Option<String>,
    /// The Gerber files' unit: `mm` or `inch`.
    pub gerber_units: Option<String>,
    /// The Gerber coordinates' decimals: 5 or 6.
    pub gerber_decimals: Option<u8>,
    /// The pick-and-place's sides: `both` (one file), `separate` (a file per side), `top` or `bottom`.
    pub pick_place_sides: Option<String>,
    /// The pick-and-place's unit: `mm` or `inch`.
    pub pick_place_units: Option<String>,
    /// The bill of materials' columns, in order: `designators`, `quantity`, `value`, `package`, `mpn`, `manufacturer`.
    pub bom_columns: Option<Vec<String>>,
    /// Parts with the same part number, value and package share a BOM line (else a line per designator).
    pub bom_group_by_value: Option<bool>,
    /// Also answer this layer of the table (`F.Mask`, `B.Paste`) drawn as SVG, as its Gerber
    /// makes it, for a preview; `Fabrication drawing` draws the fabrication drawing.
    pub preview_layer: Option<String>,
    /// Report the pre-flight checks (DRC, ERC, unrouted connections, pending schematic changes).
    #[serde(default)]
    pub preflight: bool,
    /// Answer the board's layout with these settings saved into it, for the caller to write.
    #[serde(default)]
    pub remember: bool,
    /// The Gerber job file's creation date, ISO 8601 (`2026-11-06T09:30:00Z`). Without one the
    /// job file has no date, so the same board makes the same files.
    #[serde(default)]
    pub date: Option<String>,
}

impl FabricationOptions {
    /// Every output off: the answer is only the board's layers, variants and settings.
    pub fn plan_only(&self) -> bool {
        [self.gerber, self.drill, self.drill_map, self.drawing, self.pick_place, self.bom, self.assembly_drawing, self.ipc2581, self.odb, self.ipc356]
            .iter()
            .all(|f| *f == Some(false))
    }

    /// The settings: `saved` (or the defaults) with every given field over it.
    pub fn settings(&self, saved: Option<&runmat_circuit::fab::settings::FabSettings>) -> Result<runmat_circuit::fab::settings::FabSettings, ToolError> {
        use runmat_circuit::fab::settings::{FabOutput, FabPreset};
        let mut s = saved.cloned().unwrap_or_default();
        if let Some(p) = &self.preset {
            let preset = FabPreset::from_word(p).ok_or_else(|| ToolError::invalid(format!("`{p}` is not a preset; expected fab_house, assembly, full or custom")))?;
            s.preset = preset;
            if preset != FabPreset::Custom {
                s.outputs = preset.outputs();
                s.zip = preset == FabPreset::FabHouse;
            }
        }
        let flags = [
            (self.gerber, FabOutput::Gerber),
            (self.drill, FabOutput::Drill),
            (self.drill_map, FabOutput::DrillMap),
            (self.drawing, FabOutput::Drawing),
            (self.pick_place, FabOutput::PickPlace),
            (self.bom, FabOutput::Bom),
            (self.assembly_drawing, FabOutput::AssemblyDrawing),
            (self.ipc2581, FabOutput::Ipc2581),
            (self.odb, FabOutput::Odb),
            (self.ipc356, FabOutput::Ipc356),
        ];
        let mut changed = false;
        // A plan's flags only stop the export; the settings stay as saved.
        for (flag, output) in flags.into_iter().filter(|_| !self.plan_only()) {
            match flag {
                Some(true) => changed |= s.outputs.insert(output),
                Some(false) => changed |= s.outputs.remove(&output),
                None => {}
            }
        }
        if changed {
            s.preset = FabPreset::of(&s.outputs);
        }
        if let Some(v) = self.include_through_hole {
            s.include_through_hole = v;
        }
        if let Some(v) = &self.variants {
            s.variants = v.clone();
        }
        if let Some(l) = &self.layers {
            s.layers = Some(l.iter().cloned().collect());
        }
        if let Some(f) = &self.folder {
            s.folder = Some(f.clone()).filter(|f| !f.trim().is_empty());
        }
        if let Some(z) = self.zip {
            s.zip = z;
        }
        if let Some(r) = &self.revision {
            s.revision = Some(r.trim().to_string()).filter(|r| !r.is_empty());
        }
        let unit = |u: &str| runmat_circuit::fab::LengthUnit::from_word(u).ok_or_else(|| ToolError::invalid(format!("`{u}` is not a unit; expected mm or inch")));
        if let Some(u) = &self.gerber_units {
            s.gerber.units = unit(u)?;
        }
        if let Some(d) = self.gerber_decimals {
            if d != 5 && d != 6 {
                return Err(ToolError::invalid("gerberDecimals is 5 or 6"));
            }
            s.gerber.decimals = d;
        }
        if let Some(w) = &self.pick_place_sides {
            s.pick_place.sides = runmat_circuit::fab::assembly::PlaceSides::from_word(w).ok_or_else(|| ToolError::invalid(format!("`{w}` is not a side choice; expected both, separate, top or bottom")))?;
        }
        if let Some(u) = &self.pick_place_units {
            s.pick_place.units = unit(u)?;
        }
        if let Some(cols) = &self.bom_columns {
            s.bom.columns = cols
                .iter()
                .map(|c| runmat_circuit::fab::assembly::BomColumn::from_word(c).ok_or_else(|| ToolError::invalid(format!("`{c}` is not a BOM column; expected designators, quantity, value, package, mpn or manufacturer"))))
                .collect::<Result<_, _>>()?;
        }
        if let Some(g) = self.bom_group_by_value {
            s.bom.group_by_value = g;
        }
        Ok(s)
    }
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
    /// A product family: one file per configuration (the default, then each named one) of a
    /// part or an assembly (STEP, 3MF, STL or glTF) or of the part a drawing shows (every sheet
    /// as DXF or DWG).
    #[serde(default)]
    pub all_configurations: bool,
    /// The named configuration to export (a part's or an assembly's STEP, 3MF, STL or glTF; a
    /// drawing's part).
    #[serde(default)]
    pub configuration: Option<String>,
    /// Input values for this export, by input name (`{"thickness": "3.5 mm"}`): the part or
    /// assembly regenerates with them before it is written; the file is not changed. An unknown
    /// name is an `unknown_input` error listing the declared inputs.
    #[serde(default)]
    pub set: Option<BTreeMap<String, String>>,
    /// Fabrication: which outputs to make (all by default).
    #[serde(default)]
    pub fabrication: Option<FabricationOptions>,
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
    /// Fabrication: the board, its layers and the variants the assembly files cover.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fabrication: Option<fabrication::FabricationSummary>,
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

pub(crate) fn text_file(name: String, mime: &str, content: String) -> ExportedFile {
    ExportedFile {
        bytes: content.len(),
        name,
        mime_type: mime.into(),
        encoding: "utf8".into(),
        content,
    }
}

pub(crate) fn binary_file(name: String, mime: &str, bytes: &[u8]) -> ExportedFile {
    ExportedFile {
        bytes: bytes.len(),
        name,
        mime_type: mime.into(),
        encoding: "base64".into(),
        content: base64::engine::general_purpose::STANDARD.encode(bytes),
    }
}

/// A part or assembly (in its named configuration `configuration`) as one 3D file named
/// `file_stem` plus the format's extension.
pub(crate) fn model_export(
    path: &str,
    files: &Files,
    format: ExportFormat,
    configuration: Option<&str>,
    file_stem: &str,
    notes: &mut Vec<String>,
) -> Result<Vec<ExportedFile>, ToolError> {
    let name = stem(path);
    let scene = model_scene_in(path, files, configuration)?;
    let meshes = scene.world_meshes();
    let mut note = |n: &str| {
        if !notes.iter().any(|x| x == n) {
            notes.push(n.to_owned());
        }
    };
    Ok(match format {
        ExportFormat::Step => match exact_step::exact_step_in(path, files, configuration)? {
            Some(bytes) => {
                note("STEP AP242: the exact B-rep of the regenerated bodies, with the part's semantic PMI");
                vec![text_file(format!("{file_stem}.step"), "model/step", String::from_utf8_lossy(&bytes).into_owned())]
            }
            None => {
                note("STEP AP214 faceted B-rep (the grid test kernel has no exact STEP writer), millimetres");
                vec![text_file(format!("{file_stem}.step"), "model/step", step(&meshes, &name))]
            }
        },
        ExportFormat::Stl => vec![binary_file(format!("{file_stem}.stl"), "model/stl", &stl(&meshes, &name))],
        ExportFormat::ThreeMf => {
            // UUIDs derive from the path, and a configuration's file from its name too.
            let seed = if file_stem == name { path.to_owned() } else { format!("{path}#{file_stem}") };
            vec![binary_file(format!("{file_stem}.3mf"), "model/3mf", &three_mf(&scene, &seed, &name))]
        }
        ExportFormat::Gltf => vec![binary_file(format!("{file_stem}.glb"), "model/gltf-binary", &glb(&meshes))],
        other => {
            return Err(ToolError::new("unsupported", format!("{other:?} is not a 3D model format")));
        }
    })
}

pub fn export(input: &ExportInput, files: &Files) -> Result<ExportOutput, ToolError> {
    match input.set.as_ref().filter(|set| !set.is_empty()) {
        Some(set) => export_files(input, &crate::inspect::run_set::with_inputs(&input.path, files, set)?),
        None => export_files(input, files),
    }
}

fn export_files(input: &ExportInput, files: &Files) -> Result<ExportOutput, ToolError> {
    let kind = kind_of(&input.path)?;
    let text = text_of(files, &input.path)?;
    let name = stem(&input.path);
    let mut notes = Vec::new();
    let mut fab_summary = None;
    let config = input.configuration.as_deref();
    let files_out = match input.format {
        ExportFormat::Step | ExportFormat::Stl | ExportFormat::ThreeMf | ExportFormat::Gltf
            if input.all_configurations =>
        {
            family::model_family(&input.path, files, input.format, &mut notes)?
        }
        ExportFormat::Dxf | ExportFormat::Dwg if input.all_configurations && kind == DesignKind::Drawing => {
            family::drawing_family(&input.path, files, input.format == ExportFormat::Dwg, &mut notes)?
        }
        ExportFormat::Bom => vec![family::bom_csv(&input.path, files)?],
        ExportFormat::Step | ExportFormat::Stl | ExportFormat::ThreeMf | ExportFormat::Gltf => {
            model_export(&input.path, files, input.format, config, &name, &mut notes)?
        }
        ExportFormat::Dxf => match kind {
            DesignKind::Drawing => {
                let (sheet, dxf, n) =
                    drawing_dxf(&input.path, text, files, input.sheet.as_deref(), config)?;
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
        ExportFormat::Dwg => match kind {
            DesignKind::Drawing => {
                let (sheet_name, sheet, n) = sheet::drawing_sheet(&input.path, text, files, input.sheet.as_deref(), config)?;
                notes.extend(n);
                vec![binary_file(format!("{name}-{sheet_name}.dwg"), "image/vnd.dwg", &crate::formats::dwg::write(&sheet))]
            }
            DesignKind::Part => vec![binary_file(
                format!("{name}-flat.dwg"),
                "image/vnd.dwg",
                &crate::formats::dwg::write(&sheet::flat_pattern(&input.path, text)?),
            )],
            other => {
                return Err(ToolError::new(
                    "unsupported",
                    format!("DWG exports drawings and part flat patterns; {} is a {other:?} file", input.path),
                ))
            }
        },
        ExportFormat::Pdf => match kind {
            DesignKind::Drawing => {
                let (sheet_name, sheet, n) = sheet::drawing_sheet(&input.path, text, files, input.sheet.as_deref(), config)?;
                notes.extend(n);
                vec![binary_file(format!("{name}-{sheet_name}.pdf"), "application/pdf", &crate::formats::pdf::write(&sheet))]
            }
            k if k.is_circuit() => {
                let options = crate::circuit_values::circuit_options(&input.path, text, files);
                let out = runmat_circuit::runtime::compile(&input.path, text, files, &options);
                let index = out.index.ok_or_else(|| {
                    ToolError::new(
                        "not_compiled",
                        format!("{} doesn't compile (design.check lists why): {}", input.path, out.diagnostics.first().map(|d| d.message.as_str()).unwrap_or("")),
                    )
                })?;
                let pages: Vec<_> = index.schematic.sheets.iter().map(schematic_pdf::page_sheet).collect();
                if pages.is_empty() {
                    return Err(ToolError::new("unsupported", format!("{} has no schematic pages", input.path)));
                }
                notes.push(format!("{} schematic page(s)", pages.len()));
                vec![binary_file(format!("{name}-schematic.pdf"), "application/pdf", &crate::formats::pdf::write_pages(&pages))]
            }
            other => {
                return Err(ToolError::new(
                    "unsupported",
                    format!("PDF exports drawing sheets and circuit schematics; {} is a {other:?} file", input.path),
                ))
            }
        },
        ExportFormat::Fabrication => {
            if !kind.is_circuit() {
                return Err(ToolError::new(
                    "unsupported",
                    format!("fabrication outputs come from a board; {} is a {kind:?} file", input.path),
                ));
            }
            let opts = input.fabrication.clone().unwrap_or_default();
            let (made, summary) = fabrication::fabrication(&input.path, text, files, &opts, &mut notes)?;
            fab_summary = Some(summary);
            made
        }
        ExportFormat::KicadNetlist | ExportFormat::Spice | ExportFormat::Edif | ExportFormat::Ipc356 => {
            if !kind.is_circuit() {
                return Err(ToolError::new("unsupported", format!("netlists come from circuits; {} is a {kind:?} file", input.path)));
            }
            netlist::netlist(&input.path, text, files, input.format, &name, &mut notes)?
        }
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
            .map_err(|e| match e {
                runmat_circuit::runtime::KicadExportError::NotCompiled(d) => ToolError::new(
                    "not_compiled",
                    format!(
                        "{} doesn't compile (design.check lists why): {}",
                        input.path,
                        d.first().map(|d| d.message.as_str()).unwrap_or("")
                    ),
                ),
                runmat_circuit::runtime::KicadExportError::Unsupported(why) => {
                    ToolError::new("unsupported", why)
                }
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
        fabrication: fab_summary,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImportFormat {
    Kicad,
    /// An XML project schematic (`.sch`) and/or board (`.brd`), version 6 or later.
    XmlProject,
    /// A shape-string JSON schematic and/or board (standard edition, `.json`).
    ShapeJson,
    /// Pipe-record schematics and/or a board (`.SchDoc`, `.PcbDoc`): `|RECORD=…|` text, or
    /// the binary compound files (give those as `base64`).
    PipeRecord,
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
    /// KiCad, XML project, shape-string JSON and pipe-record: the target stem (`boards/io` writes `boards/io.circuit`, `.circuit.layout` and,
    /// from KiCad, `.circuit.lock`; the other formats' packages go in `boards/io-footprints/`). STEP: the `.part` to write (`parts/housing.part`); the STEP file goes
    /// beside it.
    pub target: String,
    /// KiCad: the root schematic's file name (default: the schematic no other sheet names).
    #[serde(default)]
    pub root: Option<String>,
}

fn decode_bytes(file: &ImportFile) -> Result<Vec<u8>, ToolError> {
    match file.encoding.as_deref() {
        Some("base64") => base64::engine::general_purpose::STANDARD
            .decode(file.content.as_bytes())
            .map_err(|e| ToolError::invalid(format!("{}: not base64: {e}", file.name))),
        _ => Ok(file.content.clone().into_bytes()),
    }
}

/// Pipe-record files: ASCII ones as text, binary ones (compound files) as bytes.
fn import_pipe_record(input: &ImportInput, stem: &str) -> Result<runmat_circuit::import::Imported, String> {
    let mut text = BTreeMap::new();
    let mut binary = BTreeMap::new();
    for f in &input.files {
        let bytes = decode_bytes(f).map_err(|e| e.message)?;
        if bytes.starts_with(&[0xD0, 0xCF, 0x11, 0xE0]) {
            binary.insert(f.name.clone(), bytes);
        } else {
            let t = String::from_utf8(bytes).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());
            text.insert(f.name.clone(), t);
        }
    }
    match (text.is_empty(), binary.is_empty()) {
        (false, false) => Err("give ASCII or binary .SchDoc / .PcbDoc files, not both".into()),
        (_, false) => runmat_circuit::import::pipe_record::binary::import(&binary, stem),
        _ => runmat_circuit::import::pipe_record::import(&text, stem),
    }
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

/// What `design.import` answers: the staged change and, from a PCB-tool import, the report of
/// what it carried over, approximated and left out.
#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ImportChange {
    #[serde(flatten)]
    pub change: StagedChange,
    /// Counts carried over by kind (`carried`), and each approximation (`approximated`) and
    /// omission (`notCarried`) with how often it occurred; `format` names the format read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<serde_json::Value>")]
    pub report: Option<runmat_circuit::import::Report>,
}

pub fn import(input: &ImportInput, files: &Files) -> Result<ImportChange, ToolError> {
    let plain = |change| Ok(ImportChange { change, report: None });
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
                "Import {} project as {}",
                read.report.format,
                read.files.keys().cloned().collect::<Vec<_>>().join(", ")
            );
            let mut change = stage(summary, read.files.into_iter().collect(), files, &primary);
            change.notes.extend(read.warnings);
            Ok(ImportChange { change, report: Some(read.report) })
        }
        ImportFormat::XmlProject | ImportFormat::ShapeJson | ImportFormat::PipeRecord => {
            let stem = target.strip_suffix(".circuit.layout").or_else(|| target.strip_suffix(".circuit")).unwrap_or(&target).to_owned();
            let read = if input.format == ImportFormat::PipeRecord {
                import_pipe_record(input, &stem)
            } else {
                let mut given = BTreeMap::new();
                for f in &input.files {
                    given.insert(f.name.clone(), decode(f)?);
                }
                match input.format {
                    ImportFormat::XmlProject => runmat_circuit::import::xml_project::import(&given, &stem),
                    _ => runmat_circuit::import::shape_json::import(&given, &stem),
                }
            };
            let read = read.map_err(ToolError::invalid)?;
            let primary = read.files.keys().find(|k| k.ends_with(".circuit")).or_else(|| read.files.keys().find(|k| k.ends_with(".circuit.layout"))).cloned().unwrap_or_default();
            let summary = format!("Import {} project as {}", read.report.format, read.files.keys().filter(|k| !k.ends_with(".kicad_mod")).cloned().collect::<Vec<_>>().join(", "));
            let mut change = stage(summary, read.files.into_iter().collect(), files, &primary);
            change.notes.extend(read.report.lines());
            Ok(ImportChange { change, report: Some(read.report) })
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
            // AP242 semantic PMI in the file becomes the part's PMI on the imported faces.
            let pmi = runmat_cad::step::pmi::read(&content)
                .map(|p| runmat_cad_wasm::exchange::pmi::part_statements(&p, "step"))
                .unwrap_or_default();
            let part = format!(
                "runmat 0.7\nstd 1\n\npart {part_name} {{\n    feature step = import(source = \"{step_name}\", digest = sha256_{digest})\n{pmi}}}\n"
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
            plain(change)
        }
    }
}
