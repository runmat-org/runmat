//! The agent's command list: `docs/design/commands.md`, rendered from the catalog, so an agent
//! finds every command's id, what it does and the fields it needs without opening the Desktop.

use crate::{Catalog, CommandDescriptor, Field, FieldKind};
use std::collections::BTreeMap;

fn snake<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

fn describe(kind: &FieldKind) -> String {
    match kind {
        // A unitless quantity (a ratio) is a plain number.
        FieldKind::Quantity { unit, default, .. } => {
            let number = if unit.is_empty() {
                "number".to_owned()
            } else {
                format!("number in {unit}")
            };
            match default {
                Some(d) => format!("{number}, default {d}"),
                None => number,
            }
        }
        FieldKind::Count { default, .. } => match default {
            Some(d) => format!("whole number, default {d}"),
            None => "whole number".to_owned(),
        },
        FieldKind::Choice {
            options, default, ..
        } => {
            let values: Vec<String> = options.iter().map(|o| format!("`{}`", o.value)).collect();
            match default {
                Some(d) => format!("one of {}, default `{d}`", values.join(", ")),
                None => format!("one of {}", values.join(", ")),
            }
        }
        FieldKind::Selection {
            accepts, multiple, ..
        } => {
            let kinds: Vec<String> = accepts.iter().map(|k| k.noun().to_owned()).collect();
            format!(
                "{} {}",
                if *multiple { "picks:" } else { "pick:" },
                kinds.join(" or ")
            )
        }
        FieldKind::Boolean { default } => format!("true or false, default {default}"),
        FieldKind::Text { default, .. } => match default {
            Some(d) if !d.is_empty() => format!("text, default `{d}`"),
            _ => "text".to_owned(),
        },
        FieldKind::List { item, .. } => format!("list of {}", describe(item)),
    }
}

fn field_line(field: &Field) -> String {
    let mut line = format!("`{}` ({})", field.id, describe(&field.kind));
    if let Some(when) = &field.visible_when {
        if when.filled {
            line.push_str(&format!(" when `{}` is set", when.field));
        } else {
            line.push_str(&format!(
                " when `{}` is {}",
                when.field,
                when.any_of.join(" or ")
            ));
        }
    }
    line
}

fn command_entry(command: &CommandDescriptor) -> String {
    let mut out = format!("- `{}`: {}\n", command.id, command.description);
    let required: Vec<String> = command
        .inputs
        .iter()
        .filter(|f| f.required)
        .map(field_line)
        .collect();
    let optional: Vec<String> = command
        .inputs
        .iter()
        .filter(|f| !f.required)
        .map(field_line)
        .collect();
    if !required.is_empty() {
        out.push_str(&format!("  - Required: {}\n", required.join("; ")));
    }
    if !optional.is_empty() {
        out.push_str(&format!("  - Optional: {}\n", optional.join("; ")));
    }
    if !command.context.is_empty() {
        let kinds: Vec<String> = command
            .context
            .iter()
            .map(|k| format!("`{}`", snake(k)))
            .collect();
        out.push_str(&format!(
            "  - Runs on the `selection`: {}\n",
            kinds.join(", ")
        ));
    }
    out
}

impl Catalog {
    /// `docs/design/commands.md`: every command, grouped by id prefix, with its required and optional fields.
    pub fn commands_reference(&self) -> String {
        let mut out = String::from(
            "# Design commands\n\n\
             `design_propose_edit` runs one of these commands: give its id as `command` and its fields as `values`, \
             one per field id. A pick field takes the item's reference text or Outline id as its value, \
             for example `{\"command\": \"part.sketch\", \"values\": {\"plane\": \"origin.xy\"}}`. \
             A command that runs on existing items also takes them in `selection` (`feature:boss`, `component:R1`). \
             Numbers are in the unit shown. A field marked *when* applies only while another field has that value.\n\n\
             Plane picks take an origin plane (`origin.xy`, `origin.yz`, `origin.zx`), a datum plane (`plane.<name>`) \
             or a planar face (`feature.body.face.end`). Writing the file's text with `replace` or `text` works for any change; \
             see `design/mcad/part/features.md` and `design/ecad/circuit` for the syntax.\n\n\
             This file is generated from the command catalog.\n",
        );
        let mut groups: BTreeMap<&str, Vec<&CommandDescriptor>> = BTreeMap::new();
        for command in &self.commands {
            groups
                .entry(command.id.split('.').next().unwrap_or(&command.id))
                .or_default()
                .push(command);
        }
        for (prefix, mut commands) in groups {
            commands.sort_by(|a, b| a.id.cmp(&b.id));
            let mut editors: Vec<&str> = Vec::new();
            for editor in commands.iter().flat_map(|c| c.editors.iter()) {
                if !editors.contains(&editor.as_str()) {
                    editors.push(editor.as_str());
                }
            }
            let files: Vec<String> = editors.iter().map(|e| format!("`.{e}`")).collect();
            out.push_str(&format!(
                "\n## `{prefix}.` commands ({})\n\n",
                files.join(", ")
            ));
            for command in commands {
                out.push_str(&command_entry(command));
            }
        }
        out
    }
}
