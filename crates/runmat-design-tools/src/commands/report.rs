//! Assembly reports as results tables: `assembly.bom` (the bill of materials, top level,
//! indented or parts only, `values.type`) and `assembly.where-used` (every occurrence of the
//! selected component's file in the assembly), each in the active configuration
//! (`values.configuration`). Rows name their occurrences (`occurrence:<uuid>`), so selecting a
//! row selects them on the canvas.

use super::{failed, RunRequest};
use crate::Files;
use runmat_cad::assembly::bom::{flat_bom, indented_bom, BomItem, BomKind, BomOptions};
use runmat_cad::assembly::model::DefinitionRef;
use runmat_cad::assembly::resolve::ResolvedAssembly;
use runmat_cad::assembly::source::DesignSource;
use serde_json::{json, Value};

const COMMANDS: &[&str] = &["assembly.bom", "assembly.where-used"];

/// Whether `id` is a report this module runs.
pub fn handles(id: &str) -> bool {
    COMMANDS.contains(&id)
}

fn file_name(definition: &DefinitionRef) -> String {
    let file = definition.file.as_str();
    let name = file.rsplit('/').next().unwrap_or(file);
    match &definition.part {
        Some(part) => format!("{name} · {part}"),
        None => name.to_owned(),
    }
}

fn material(item: &BomItem) -> String {
    item.material
        .as_ref()
        .map(|m| runmat_cad::drawing::bom_table::material_name(m.as_str()))
        .unwrap_or_else(|| "—".into())
}

fn kind_label(kind: &BomKind) -> &'static str {
    match kind {
        BomKind::Part => "Part",
        BomKind::Assembly => "Assembly",
        BomKind::Circuit => "Board",
        BomKind::Fastener => "Fastener",
    }
}

fn row(cells: Vec<String>, items: Vec<String>) -> Value {
    json!({ "cells": cells, "items": items })
}

fn table(
    title: &str,
    summary: String,
    columns: &[&str],
    rows: Vec<Value>,
    data: Value,
    settings: Vec<String>,
) -> Value {
    json!({
        "ok": true,
        "title": title,
        "summary": summary,
        "columns": columns,
        "rows": rows,
        "problems": [],
        "data": data,
        "settings": settings,
    })
}

/// The occurrences under `parent` (a path) whose file and configuration are `item`'s.
fn members(tree: &ResolvedAssembly, parent: &str, item: &BomItem) -> Vec<String> {
    tree.nodes
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, n)| {
            n.definition == item.definition
                && n.parent.is_some_and(|p| tree.path(p) == parent)
                && (item.configuration.is_empty() || n.configuration.label() == item.configuration)
        })
        .map(|(_, n)| format!("occurrence:{}", n.uuid))
        .collect()
}

/// Runs the report `request.command` on the assembly `path`.
pub fn run(path: &str, text: &str, files: &Files, request: &RunRequest) -> Value {
    if !path.ends_with(".assembly") {
        return failed(format!(
            "{} reports on an assembly; {path} is not one",
            request.command
        ));
    }
    let configuration = request
        .values
        .get("configuration")
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty() && *c != "default");
    let resolution =
        match runmat_cad_wasm::runtime::resolve_assembly_in(path, text, files, configuration) {
            Ok(r) => r,
            Err(d) => {
                return failed(format!(
                    "{path} doesn't compile: {}",
                    d.first()
                        .map(|d| d.message.as_str())
                        .unwrap_or("syntax error")
                ))
            }
        };
    let tree = &resolution.resolved;
    let source: &dyn DesignSource = &resolution.source;
    let mut settings = Vec::new();
    if let Some(c) = configuration {
        settings.push(format!("Configuration: {c}"));
    }
    match request.command.as_str() {
        "assembly.bom" => bom(tree, source, request, settings),
        "assembly.where-used" => where_used(tree, request, settings),
        other => {
            super::mark_unhandled();
            failed(format!("`{other}` has no handler for this file"))
        }
    }
}

fn bom(
    tree: &ResolvedAssembly,
    source: &dyn DesignSource,
    request: &RunRequest,
    mut settings: Vec<String>,
) -> Value {
    let kind = request
        .values
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("top");
    let columns = [
        "Item",
        "Part number",
        "File",
        "Type",
        "Configuration",
        "Material",
        "Qty",
    ];
    let (label, rows, data) = match kind {
        "parts" => {
            let bom = flat_bom(tree, Some(source), &BomOptions::default());
            let path_ids: std::collections::BTreeMap<String, String> = tree
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (tree.path(i as u32), format!("occurrence:{}", n.uuid)))
                .collect();
            let rows = bom
                .rows
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let items = r
                        .paths
                        .iter()
                        .filter_map(|p| path_ids.get(p).cloned())
                        .collect();
                    row(
                        vec![
                            (i + 1).to_string(),
                            r.item.part_number.clone().unwrap_or_else(|| "—".into()),
                            file_name(&r.item.definition),
                            kind_label(&r.item.kind).into(),
                            r.item.configuration.clone(),
                            material(&r.item),
                            r.quantity.to_string(),
                        ],
                        items,
                    )
                })
                .collect::<Vec<_>>();
            let data = serde_json::to_value(&bom).unwrap_or(Value::Null);
            ("Parts only", rows, data)
        }
        "top" | "indented" => {
            let all = indented_bom(tree, Some(source));
            let shown: Vec<_> = all
                .iter()
                .filter(|r| kind == "indented" || r.level == 1)
                .collect();
            let mut numbers: Vec<u32> = Vec::new();
            let rows = shown
                .iter()
                .map(|r| {
                    // Item numbers by level: 1, 2, 2.1, 2.2, 3.
                    let depth = r.level.max(1) as usize;
                    numbers.truncate(depth);
                    while numbers.len() < depth {
                        numbers.push(0);
                    }
                    numbers[depth - 1] += 1;
                    let item = numbers
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(".");
                    row(
                        vec![
                            item,
                            r.item.part_number.clone().unwrap_or_else(|| "—".into()),
                            file_name(&r.item.definition),
                            kind_label(&r.item.kind).into(),
                            r.item.configuration.clone(),
                            material(&r.item),
                            r.quantity.to_string(),
                        ],
                        members(tree, &r.parent, &r.item),
                    )
                })
                .collect::<Vec<_>>();
            let data = serde_json::to_value(&shown).unwrap_or(Value::Null);
            (
                if kind == "top" {
                    "Top level"
                } else {
                    "Indented"
                },
                rows,
                data,
            )
        }
        other => {
            return failed(format!(
                "the BOM type is top, indented or parts, not `{other}`"
            ))
        }
    };
    settings.insert(0, format!("Type: {label}"));
    let total: u64 = tree
        .nodes
        .iter()
        .skip(1)
        .filter(|n| n.definition.kind != runmat_cad::assembly::model::DefinitionKind::Assembly)
        .count() as u64;
    let summary = format!("{} rows · {total} parts in the assembly", rows.len());
    table("BOM", summary, &columns, rows, data, settings)
}

fn where_used(tree: &ResolvedAssembly, request: &RunRequest, settings: Vec<String>) -> Value {
    // A face or edge picked on the canvas (`occurrence:<uuid>/face:0:4`) names its component.
    let picked: Vec<&str> = request
        .selection
        .iter()
        .filter_map(|id| id.strip_prefix("occurrence:"))
        .map(|id| id.split('/').next().unwrap_or(id))
        .collect();
    let [uuid] = picked.as_slice() else {
        return failed("Select one component to list where its file is used");
    };
    let Some(node) = tree.nodes.iter().find(|n| n.uuid.to_string() == *uuid) else {
        return failed("the selected component isn't in this assembly");
    };
    let definition = node.definition.clone();
    let uses = runmat_cad::assembly::bom::where_used_in(tree, &definition);
    let rows: Vec<Value> = uses
        .iter()
        .map(|&i| {
            let n = tree.node(i);
            let parent = n.parent.map(|p| tree.node(p));
            let within = match parent {
                Some(p) if p.parent.is_some() => file_name(&p.definition),
                _ => "This assembly".into(),
            };
            row(
                vec![tree.path(i), within, n.configuration.label()],
                vec![format!("occurrence:{}", n.uuid)],
            )
        })
        .collect();
    let summary = format!(
        "{} is used {} time{} in this assembly",
        file_name(&definition),
        rows.len(),
        if rows.len() == 1 { "" } else { "s" }
    );
    let data = json!({
        "definition": definition,
        "occurrences": uses.iter().map(|&i| tree.path(i)).collect::<Vec<_>>(),
    });
    table(
        "Where-used",
        summary,
        &["Occurrence", "In", "Configuration"],
        rows,
        data,
        settings,
    )
}
