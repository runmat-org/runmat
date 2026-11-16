//! `circuit.net-length`: the selected nets' lengths on the board, as a results table. Routed
//! is [`crate::route::net_length`] (tracks plus each via's barrel through the stackup); unrouted is the straight
//! length of the net's open connections (its airwires); Limit is the net's `length_max` rule,
//! when one applies, and the row is an error past it.

use super::edit::{EditError, EditOutput, EditRequest};
use super::*;
use crate::board_rules::{props, BItem, BoardGraph, BoardRules};
use serde_json::json;

fn mm_text(nm: f64) -> String {
    format!("{:.2} mm", nm / 1.0e6)
}

pub(super) fn run(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let nets: Vec<&str> = request
        .selection
        .iter()
        .filter_map(|id| id.strip_prefix("net:"))
        .collect();
    if nets.is_empty() {
        return Err(EditError::new("select the nets to measure"));
    }
    let layout = project.layout_or_err()?;
    let board = layout.parsed.board.as_ref().expect("read");
    let (rules, _) = BoardRules::new(&board.limits, &board.rules);
    let graph = BoardGraph::new(board, &[]);
    let mut rules = rules;
    rules.rebuild(&graph);
    let airwires = crate::route::ratsnest(board);
    let mut rows = Vec::new();
    let mut over = 0;
    for name in &nets {
        let Some(i) = board.nets.iter().position(|n| n.name == *name) else {
            rows.push(json!({
                "cells": [name, "—", "—", "—", "—", "not on the board"],
                "items": [format!("net:{name}")],
                "severity": "warning",
            }));
            continue;
        };
        let net = crate::layout::NetId(i as u32);
        let vias = board.vias.values().filter(|v| v.net == Some(net)).count();
        let open: Vec<_> = airwires.iter().filter(|a| a.net == net).collect();
        let unrouted: f64 = open.iter().map(|a| a.length).sum();
        let routed = crate::route::net_length(board, net);
        let limit = rules.len(&graph, BItem::Net(i as u32), props::LENGTH_MAX);
        let past = limit.is_some_and(|l| routed + unrouted > l as f64);
        if past {
            over += 1;
        }
        let status = match (open.len(), past) {
            (_, true) => "over its limit".to_owned(),
            (0, false) => "routed".to_owned(),
            (n, false) => format!("{n} open"),
        };
        let mut row = json!({
            "cells": [
                name,
                mm_text(routed),
                mm_text(unrouted),
                vias.to_string(),
                limit.map_or("—".to_owned(), |l| mm_text(l as f64)),
                status,
            ],
            "items": [format!("net:{name}")],
        });
        if past {
            row["severity"] = json!("error");
        }
        rows.push(row);
    }
    let summary = match (nets.len(), over) {
        (1, 0) => format!("{} on {}", nets[0], layout.path),
        (n, 0) => format!("{n} nets on {}", layout.path),
        (n, o) => format!("{n} nets on {}; {o} over its length limit", layout.path),
    };
    Ok(EditOutput {
        files: Vec::new(),
        summary: summary.clone(),
        report: Some(json!({
            "title": "Net length",
            "summary": summary,
            "columns": ["Net", "Routed", "Unrouted", "Vias", "Limit", "Status"],
            "rows": rows,
            "problems": [],
        })),
    })
}
