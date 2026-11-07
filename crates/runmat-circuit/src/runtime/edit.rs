//! Catalog commands on a circuit: board edits, ECO in both directions, room formats; each a
//! minimal text change per file.

use crate::layer_set::CopperIndex;
use super::*;

/* ------------------------------------------------------------------------------------------------
 * Edits
 * --------------------------------------------------------------------------------------------- */

/// A catalog command's run, as the Desktop sends it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EditRequest {
    pub command: String,
    #[serde(default)]
    pub values: serde_json::Map<String, Value>,
    /// Outline node ids (`component:R1`, `track:17`, `room:ch[0]`).
    #[serde(default)]
    pub selection: Vec<String>,
    #[serde(default)]
    pub options: CompileOptions,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EditOutput {
    pub files: Vec<FileChange>,
    /// One line on what the edit did.
    pub summary: String,
}

/// Why an edit did nothing: the files don't compile, or the edit can't be made.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EditError {
    pub message: String,
    pub diagnostics: Vec<Diagnostic>,
}

impl EditError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        EditError {
            message: message.into(),
            diagnostics: Vec::new(),
        }
    }
}

impl From<String> for EditError {
    fn from(m: String) -> Self {
        EditError::new(m)
    }
}

pub(super) fn failure(f: EditFailure) -> EditError {
    EditError::new(f.to_string())
}

/// The board commands that need the whole project (the circuit with the board: ECO, rooms,
/// the routers and the board tools), so a session runs them through [`edit`], not on its kept
/// board.
pub(super) fn needs_project(command: &str) -> bool {
    matches!(
        command,
        "board.update-schematic"
            | "board.copy-room-formats"
            | "board.fanout"
            | "board.autoroute"
            | "board.route-net"
            | "board.stitch"
            | "board.teardrops"
            | "board.diff-pair"
            | "board.tune-length"
    )
}

/// Runs a catalog command on the circuit `path` belongs to, as one change per file.
pub fn edit(
    path: &str,
    text: &str,
    files: &Files,
    request: &EditRequest,
) -> Result<EditOutput, EditError> {
    let project = load(path, text, files, &request.options).map_err(|diagnostics| EditError {
        message: format!(
            "the circuit has {} error(s), so the edit wasn't made",
            diagnostics.len()
        ),
        diagnostics,
    })?;
    match request.command.as_str() {
        "schematic.update-board" => update_board(&project, request),
        "board.update-schematic" => update_schematic(&project, request),
        "board.copy-room-formats" => room_formats(&project, request),
        "board.fanout" | "board.autoroute" | "board.route-net" => {
            super::autoroute::run(&project, request)
        }
        "board.stitch" => super::stitch::run(&project, request),
        "board.teardrops" => super::teardrops::run(&project, request),
        "board.diff-pair" => super::coupled::diff_pair(&project, request),
        "board.tune-length" => super::coupled::tune_length(&project, request),
        "schematic.annotate" => super::annotate::run(&project, request),
        "schematic.move" => super::annotate::move_items(&project, request),
        "library.place" => super::library_edit::place(&project, request),
        "circuit.set-property" => super::set_property::run(&project, request),
        c if super::schematic_edit::COMMANDS.contains(&c) => super::schematic_edit::run(&project, files, request),
        "simulate.probe" | "simulate.probe-net" | "simulate.probe-pin" | "simulate.monte-carlo" | "simulate.op-point" | "simulate.dc-sweep" | "simulate.transient" | "simulate.ac" | "simulate.noise" | "simulate.temperature" | "simulate.set-tolerance" => super::sim_edit::run(&project, request),
        command => {
            let layout = project.layout_or_err()?;
            let board = layout.parsed.board.as_ref().expect("read");
            let edit = board_edit(board, &layout.parsed.sites, command, request)?;
            let change =
                board_edit::board_command(&layout.path, &layout.text, &edit).map_err(failure)?;
            let mut summary = format!("{} on {}", edit.command(), layout.path);
            if let (BoardEdit::LayerStack { layers, thickness, dielectric }, Some(target)) =
                (&edit, number(&request.values, "target_z"))
            {
                let stackup = board_edit::standard_stackup(*layers, *thickness, dielectric.as_ref());
                summary.push_str(&format!("; {}", widths_for(&stackup, target)));
            }
            Ok(EditOutput {
                summary,
                files: vec![file_change(&layout.path, &layout.text, change)],
            })
        }
    }
}

/// The lock file rewritten to `lock`, when it differs from the one on disk (or, with
/// `write_missing`, when there is none on disk yet).
pub(super) fn lock_change(
    project: &Project,
    lock: &LockSection,
    write_missing: bool,
) -> Option<FileChange> {
    let lock_text = lock.to_text();
    let changed = match &project.lock_text {
        Some(old) => *old != lock_text && *lock != project.lock,
        None => write_missing || *lock != project.lock,
    };
    if !changed {
        return None;
    }
    let old = project.lock_text.clone().unwrap_or_default();
    Some(FileChange {
        edits: vec![Utf16Edit {
            start: 0,
            end: utf16_offset(&old, old.len()),
            text: lock_text.clone(),
        }],
        path: lock_path_for(&project.circuit_path),
        text: lock_text,
        reprinted: true,
    })
}

/// The ECO lines a run selected: its `changes` value holds Outline node ids
/// (`eco:<line>`, `eco-back:<line>`), as a list or as one text with the ids separated by
/// commas (an agent's proposal gives each field one text). Line ids hold no commas.
fn selected_lines(values: &serde_json::Map<String, Value>, prefix: &str) -> BTreeSet<String> {
    strings(values.get("changes"))
        .iter()
        .flat_map(|text| text.split(','))
        .filter_map(|id| id.trim().strip_prefix(prefix).map(str::to_owned))
        .collect()
}

fn strings(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn flag(values: &serde_json::Map<String, Value>, id: &str, default: bool) -> bool {
    values.get(id).and_then(Value::as_bool).unwrap_or(default)
}

fn number(values: &serde_json::Map<String, Value>, id: &str) -> Option<f64> {
    values.get(id).and_then(Value::as_f64)
}

/// All or nothing: the selected lines that fail validation, named.
fn refuse(plan: &EcoPlan, board: &Board, selected: &BTreeSet<String>) -> Result<(), EditError> {
    if selected.is_empty() {
        return Err(EditError::new("select at least one change to apply"));
    }
    let unknown: Vec<&String> = selected
        .iter()
        .filter(|id| !plan.changes.iter().any(|c| &c.id == *id))
        .collect();
    if !unknown.is_empty() {
        return Err(EditError::new(format!(
            "{} selected change(s) are no longer pending: {}",
            unknown.len(),
            unknown
                .iter()
                .take(3)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    let checks = eco::validate(plan, board, selected);
    let failures: Vec<String> = plan
        .changes
        .iter()
        .filter(|c| checks.get(&c.id).is_some_and(|k| k.status == Status::Error))
        .map(|c| {
            format!(
                "{} ({})",
                c.item,
                checks[&c.id].reason().unwrap_or_default()
            )
        })
        .collect();
    if failures.is_empty() {
        return Ok(());
    }
    Err(EditError::new(format!(
        "{} {} validation, so nothing was applied: {}{}",
        failures.len(),
        if failures.len() == 1 {
            "change fails"
        } else {
            "changes fail"
        },
        failures
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("; "),
        if failures.len() > 3 { "…" } else { "" }
    )))
}

fn update_board(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let board = project.board().expect("read");
    let run = EcoSettings::from_values(&request.values);
    let (plan, _) = plans(project, board, Some(&run));
    let selected = selected_lines(&request.values, "eco:");
    refuse(&plan, board, &selected)?;
    let options = ApplyOptions {
        remove_orphaned_copper: flag(&request.values, "remove_orphaned_copper", false),
    };
    let (change, report) = board_edit::apply_eco(
        &layout.path,
        &layout.text,
        &plan,
        &selected,
        &project.pool,
        &options,
    )
    .map_err(failure)?;
    Ok(EditOutput {
        summary: eco_summary(&plan, &report),
        files: vec![file_change(&layout.path, &layout.text, change)],
    })
}

/// What an applied ECO did, in a sentence a reviewer reads: the one change named
/// (`C900: C_0402_1005Metric_IPCN → C_0805_2012Metric_IPCN`), or the count, then the copper it
/// removed and any copper left on no net.
fn eco_summary(plan: &eco::EcoPlan, report: &eco::ApplyReport) -> String {
    let count =
        |n: usize, one: &str, many: &str| format!("{n} {}", if n == 1 { one } else { many });
    let mut out = match report.applied.as_slice() {
        [id] => match plan.get(id) {
            Some(c) => match (&c.from, &c.to) {
                (Some(from), Some(to)) => format!("Applied {}: {from} → {to}.", c.item),
                _ => format!("Applied the change to {}.", c.item),
            },
            None => "Applied 1 change.".to_string(),
        },
        all => {
            // One edit made to several parts (a sheet's part in every channel): the parts, then
            // each kind of change, when every part had the same changes.
            let changes: Vec<_> = all.iter().filter_map(|id| plan.get(id)).collect();
            // Each edit (from → to) and the parts it reached, by designator (a parameter row
            // names `C4 capacitance`).
            let mut kinds: Vec<(String, String, BTreeSet<&str>)> = Vec::new();
            let mut parts: Vec<&str> = Vec::new();
            for c in &changes {
                let part = c.item.split(' ').next().unwrap_or_default();
                if !parts.contains(&part) {
                    parts.push(part);
                }
                let (Some(from), Some(to)) = (&c.from, &c.to) else {
                    continue;
                };
                match kinds.iter_mut().find(|(f, t, _)| f == from && t == to) {
                    Some((_, _, at)) => {
                        at.insert(part);
                    }
                    None => kinds.push((from.clone(), to.clone(), [part].into())),
                }
            }
            let items = parts;
            let alike = changes.len() == all.len()
                && !kinds.is_empty()
                && kinds.iter().all(|k| k.2.len() == items.len());
            if alike && !items.is_empty() && items.len() <= 6 {
                let edits: Vec<String> =
                    kinds.iter().map(|(f, t, _)| format!("{f} → {t}")).collect();
                format!("Applied {}: {}.", items.join(", "), edits.join(", "))
            } else {
                format!("Applied {}.", count(all.len(), "change", "changes"))
            }
        }
    };
    let (tracks, vias) = (report.removed_tracks.len(), report.removed_vias.len());
    out.push_str(&match (tracks, vias) {
        (0, 0) => " No tracks or vias were removed.".to_string(),
        (t, 0) => format!(" Removed {}.", count(t, "track", "tracks")),
        (0, v) => format!(" Removed {}.", count(v, "via", "vias")),
        (t, v) => format!(
            " Removed {} and {}.",
            count(t, "track", "tracks"),
            count(v, "via", "vias")
        ),
    });
    if !report.unnetted.is_empty() {
        out.push_str(&format!(
            " {} left on no net.",
            count(report.unnetted.len(), "copper item is", "copper items are")
        ));
    }
    out
}

fn update_schematic(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let board = project.board().expect("read");
    let plan = eco::back_annotate(&project.schematic(), board, &board.sync.options);
    let selected = selected_lines(&request.values, "eco-back:");
    refuse(&plan, board, &selected)?;
    let mut design = project.design.clone();
    let mut lock = project.lock.clone();
    let (board_change, applied) = board_edit::edit_board(&layout.path, &layout.text, |b| {
        eco::apply_back(&mut design, &mut lock, b, &plan, &selected).map_err(|e| format!("{e:?}"))
    })
    .map_err(failure)?;
    // The file's own sheets take the change; a change in a used file is not written here.
    let mut own_new = project.own.clone();
    for (name, sheet) in own_new.sheets.iter_mut() {
        if let Some(changed) = design.sheets.get(name) {
            *sheet = changed.clone();
        }
    }
    for (name, sheet) in &design.sheets {
        if !project.own.sheets.contains_key(name) && project.design.sheets.get(name) != Some(sheet)
        {
            return Err(EditError::new(format!(
                "sheet `{name}` comes from a used file; back-annotate there"
            )));
        }
    }
    let mut files = vec![file_change(&layout.path, &layout.text, board_change)];
    if own_new != project.own {
        let change = diff_circuit(
            &project.circuit_path,
            &project.circuit_text,
            &project.own,
            &own_new,
        )?;
        files.push(file_change(
            &project.circuit_path,
            &project.circuit_text,
            change,
        ));
    }
    files.extend(lock_change(project, &lock, false));
    Ok(EditOutput {
        summary: format!("back-annotated {} change(s)", applied.len()),
        files,
    })
}

/// `room:ch[0]` → `ch[0]`; a bare room name stays.
fn room_name(id: &str) -> String {
    id.strip_prefix("room:").unwrap_or(id).to_owned()
}

fn room_formats(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let source = strings(request.values.get("source"))
        .into_iter()
        .chain(
            request
                .selection
                .iter()
                .filter(|s| s.starts_with("room:"))
                .cloned(),
        )
        .next()
        .map(|s| room_name(&s))
        .ok_or_else(|| EditError::new("pick the room to copy from"))?;
    let targets: Vec<String> = strings(request.values.get("targets"))
        .iter()
        .map(|s| room_name(s))
        .filter(|t| t != &source)
        .collect();
    let options = RoomFormatOptions {
        placement: flag(&request.values, "placement", true),
        routing: flag(&request.values, "routing", true),
        room_shape: flag(&request.values, "room_shape", true),
        pours: flag(&request.values, "pours", true),
        rotate: flag(&request.values, "rotate", false),
    };
    let (change, report) =
        board_edit::copy_room_formats(&layout.path, &layout.text, &source, &targets, &options)
            .map_err(failure)?;
    let placed: usize = report.copies.iter().map(|c| c.placed.len()).sum();
    let tracks: usize = report.copies.iter().map(|c| c.tracks_added.len()).sum();
    let pours: usize = report.copies.iter().map(|c| c.pours_added.len()).sum();
    Ok(EditOutput {
        summary: format!(
            "copied {source} into {} room(s): {placed} part(s) placed, {tracks} track(s) and {pours} pour(s) copied",
            report.copies.len()
        ),
        files: vec![file_change(&layout.path, &layout.text, change)],
    })
}

/// An Outline node id or designator → the site or designator `apply_board_edit` resolves.
fn site_of(board: &Board, sites: &BTreeMap<ItemId, String>, id: &str) -> Result<String, EditError> {
    let (kind, rest) = id.split_once(':').unwrap_or(("", id));
    match kind {
        "component" | "footprint" | "" => {
            if board.footprint_by_ref(rest).is_some() {
                Ok(rest.to_owned())
            } else {
                Err(EditError::new(format!("{rest} is not on the board")))
            }
        }
        "track" | "via" | "pour" | "zone" | "fiducial" | "text" | "keepout" | "dimension" => {
            let n: u64 = rest
                .parse()
                .map_err(|_| EditError::new(format!("`{id}` is not a board item")))?;
            sites
                .get(&ItemId(n))
                .cloned()
                .ok_or_else(|| EditError::new(format!("`{id}` is not on the board")))
        }
        _ => Err(EditError::new(format!("`{id}` is not a board item"))),
    }
}

const MM: f64 = 1_000_000.0;

/// The `material` input of `board.layer-stack`: FR-4 (the standard cores and prepreg, `None`)
/// or polyimide for every dielectric layer.
fn dielectric(name: Option<&str>) -> Result<Option<crate::layout::Material>, EditError> {
    let Some(name) = name.map(str::trim).filter(|n| !n.is_empty()) else { return Ok(None) };
    let key: String = name.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
    match key.as_str() {
        "fr4" => Ok(None),
        "polyimide" => Ok(Some(crate::layout::Material::polyimide())),
        _ => Err(EditError::new(format!(
            "`{name}` is not a dielectric the layer stack knows; use FR-4 or Polyimide, or write the material in the stackup"
        ))),
    }
}

/// The track width that meets `target` Ω on each copper layer of `stackup`, as one line:
/// `50 Ω: F.Cu 0.294 mm (microstrip), In1.Cu 0.127 mm (stripline), …`.
fn widths_for(stackup: &crate::layout::Stackup, target: f64) -> String {
    use crate::impedance::TraceGeometry;
    let names = stackup.copper_names();
    let rows: Vec<String> = names
        .iter()
        .enumerate()
        .map(|(i, name)| match TraceGeometry::of(stackup, i as CopperIndex) {
            Some(g) => {
                let kind = match g {
                    TraceGeometry::Microstrip { .. } => "microstrip",
                    TraceGeometry::Stripline { .. } => "stripline",
                };
                match g.width_for(target) {
                    Some(w) => format!("{name} {:.3} mm ({kind})", w as f64 / MM),
                    None => format!("{name} no width meets it ({kind})"),
                }
            }
            None => format!("{name} has no reference plane"),
        })
        .collect();
    format!("{target} Ω: {}", rows.join(", "))
}

fn mm_value(values: &serde_json::Map<String, Value>, id: &str) -> Option<crate::geom::Coord> {
    number(values, id).map(|v| (v * MM).round() as crate::geom::Coord)
}

/// The model edit a board catalog command makes, from its values and the selection.
pub(super) fn board_edit(
    board: &Board,
    sites: &BTreeMap<ItemId, String>,
    command: &str,
    request: &EditRequest,
) -> Result<BoardEdit, EditError> {
    let values = &request.values;
    let items = |field: &str| -> Result<Vec<String>, EditError> {
        let picked = strings(values.get(field));
        let ids = if picked.is_empty() {
            request.selection.clone()
        } else {
            picked
        };
        if ids.is_empty() {
            return Err(EditError::new("select the items first"));
        }
        ids.iter().map(|id| site_of(board, sites, id)).collect()
    };
    match command {
        "board.move" => Ok(BoardEdit::Move {
            items: items("items")?,
            dx: mm_value(values, "dx").unwrap_or(0),
            dy: mm_value(values, "dy").unwrap_or(0),
            rotation: number(values, "rotation").unwrap_or(0.0),
        }),
        "board.flip" => Ok(BoardEdit::Flip {
            items: items("components")?,
        }),
        "board.lock" => {
            let items = items("items")?;
            // Lock toggles: all locked unlocks, otherwise everything locks.
            let all_locked = items.iter().all(|s| {
                board.footprint_by_ref(s).is_some_and(|f| f.locked)
                    || sites.iter().find(|(_, x)| *x == s).is_some_and(|(id, _)| {
                        board.tracks.get(id).is_some_and(|t| t.locked)
                            || board.vias.get(id).is_some_and(|v| v.locked)
                            || board.zones.get(id).is_some_and(|z| z.locked)
                    })
            });
            Ok(BoardEdit::Lock {
                items,
                locked: !all_locked,
            })
        }
        "board.delete" | "edit.delete" => Ok(BoardEdit::Delete {
            items: items("items")?,
        }),
        "board.capability" => {
            let word = values.get("capability").and_then(Value::as_str).unwrap_or("standard");
            crate::capability::Capability::from_word(word)
                .map(BoardEdit::Capability)
                .ok_or_else(|| EditError::new(format!("`{word}` is not a board-house capability; expected standard, advanced or hdi")))
        }
        "board.layer-stack" => Ok(BoardEdit::LayerStack {
            layers: number(values, "layers").map_or(board.copper_count(), |n| n as CopperIndex),
            thickness: mm_value(values, "thickness").unwrap_or_else(|| board.stackup.thickness()),
            dielectric: dielectric(values.get("material").and_then(Value::as_str))?,
        }),
        "board.test-points" => {
            let nets = strings(values.get("nets"));
            let net = nets
                .first()
                .map(|n| n.strip_prefix("net:").unwrap_or(n).to_owned())
                .ok_or_else(|| EditError::new("pick the net for the test point"))?;
            if nets.len() > 1 {
                return Err(EditError::new("test points are placed one net at a time"));
            }
            let at = board
                .outline
                .outer
                .iter()
                .fold(None::<crate::geom::Point>, |acc, p| {
                    Some(acc.map_or(*p, |a| crate::geom::Point::new(a.x.min(p.x), a.y.min(p.y))))
                })
                .ok_or_else(|| {
                    EditError::new("the board has no outline to place a test point in")
                })?;
            Ok(BoardEdit::TestPoint {
                net,
                at: crate::geom::Point::new(
                    at.x + crate::geom::mm(2.0),
                    at.y + crate::geom::mm(2.0),
                ),
                side: if values.get("side").and_then(Value::as_str) == Some("top") {
                    Side::Top
                } else {
                    Side::Bottom
                },
                diameter: mm_value(values, "diameter").unwrap_or(crate::geom::mm(1.0)),
            })
        }
        "board.route" | "board.place-via" | "board.pour" | "board.keepout" | "board.room"
        | "board.dimension" | "board.outline" | "board.fill" => {
            crate::drawn::drawn_edit(board, command, values, &request.selection)
                .map_err(EditError::new)
        }
        other => Err(EditError::new(format!(
            "{other} isn't a model edit the circuit runtime runs"
        ))),
    }
}
