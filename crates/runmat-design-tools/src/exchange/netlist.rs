//! `design.export` netlists of a circuit: KiCad (`.net`), SPICE (`.cir`), EDIF 2.0.0 (`.edn`)
//! and, from its board, IPC-D-356A (`.ipc`).

use super::{text_file, ExportFormat, ExportedFile};
use crate::{Files, ToolError};

fn compile_error(path: &str, d: Vec<runmat_circuit::runtime::Diagnostic>) -> ToolError {
    ToolError::new(
        "not_compiled",
        format!(
            "{path} doesn't compile (design.check lists why): {}",
            d.first().map(|d| d.message.as_str()).unwrap_or("")
        ),
    )
}

/// The netlist of `format` for the circuit `path` belongs to, named `<stem>.<ext>`.
pub fn netlist(
    path: &str,
    text: &str,
    files: &Files,
    format: ExportFormat,
    stem: &str,
    notes: &mut Vec<String>,
) -> Result<Vec<ExportedFile>, ToolError> {
    let options = runmat_circuit::runtime::CompileOptions::default();
    if format == ExportFormat::Ipc356 {
        let request = runmat_circuit::fab::set::FabRequest {
            gerber: false,
            drill: false,
            drill_map: false,
            drawing: false,
            pick_place: false,
            bom: false,
            ipc2581: false,
            odb: false,
            ipc356: true,
            assembly_drawing: false,
            ..Default::default()
        };
        let out = runmat_circuit::runtime::export_fabrication(
            path,
            text,
            files,
            &options,
            &request,
            Some(&[]),
        )
        .map_err(|d| compile_error(path, d))?;
        return Ok(out
            .files
            .into_iter()
            .map(|(p, t)| {
                text_file(
                    p.rsplit('/').next().unwrap_or(&p).to_owned(),
                    "text/plain",
                    t,
                )
            })
            .collect());
    }
    let (nl, pool) = runmat_circuit::runtime::netlist_and_pool(path, text, files, &options)
        .map_err(|d| compile_error(path, d))?;
    Ok(vec![match format {
        ExportFormat::KicadNetlist => text_file(
            format!("{stem}.net"),
            "application/x-kicad-netlist",
            runmat_circuit::netlists::kicad(&nl, &pool),
        ),
        ExportFormat::Edif => text_file(
            format!("{stem}.edn"),
            "application/edif",
            runmat_circuit::netlists::edif(&nl, &pool),
        ),
        ExportFormat::Spice => {
            let opts = runmat_circuit_spice::netlist::EmitOptions {
                title: nl.design.clone(),
                sense: runmat_circuit_spice::netlist::SensePins::None,
                skip_unmodelled: true,
                ..Default::default()
            };
            let deck = runmat_circuit_spice::netlist::emit(&nl, &pool, &opts).map_err(|e| {
                ToolError::new(
                    "unsupported",
                    format!(
                        "{path}: {}",
                        e.first().map(|e| format!("{e:?}")).unwrap_or_default()
                    ),
                )
            })?;
            let modelled: std::collections::BTreeSet<&str> = deck
                .elements
                .iter()
                .filter_map(|e| e.designator.as_deref())
                .collect();
            let left: Vec<&str> = nl
                .components
                .iter()
                .filter(|c| c.fitted() && !modelled.contains(c.designator.as_str()))
                .map(|c| c.designator.as_str())
                .collect();
            if !left.is_empty() {
                notes.push(format!(
                    "no SPICE model, left out of the deck: {}",
                    left.join(", ")
                ));
            }
            text_file(format!("{stem}.cir"), "application/x-spice", deck.text())
        }
        other => {
            return Err(ToolError::new(
                "unsupported",
                format!("{other:?} is not a netlist format"),
            ))
        }
    }])
}
