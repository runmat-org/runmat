//! Parts go into an assembly through `design.seat_part`. A proposed edit to an assembly that
//! already places parts is refused when it adds, by hand, what a seating writes: a part
//! occurrence (`use part … as name`), a connector, or a fasten, coincident (`planar`) or
//! concentric (`cylindrical`) mate. The refusal names what the edit adds and gives an example
//! seating call.
//!
//! Allowed:
//! - changing values on existing occurrences, connectors and mates;
//! - removing them;
//! - a new assembly, or the first parts of one that places none yet (the base part is placed by
//!   hand and the rest seated on it);
//! - subassembly occurrences (`use assembly`), which a seating doesn't insert;
//! - mates of the kinds a seating doesn't write (revolute, slider, gear, distance, …), with the
//!   connectors they join.

use crate::ToolError;
use std::collections::BTreeSet;

/// The mate kinds `design.seat_part` writes.
const SEATED_KINDS: [&str; 3] = ["fasten", "planar", "cylindrical"];

#[derive(Default)]
struct Declared {
    parts: BTreeSet<String>,
    connectors: BTreeSet<String>,
    /// Each mate's name and kind (empty when the text doesn't say).
    mates: Vec<(String, String)>,
}

fn name_after(rest: &str) -> Option<String> {
    let name: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// The declarations at the start of each line: part occurrences, connectors and mates.
fn declared(text: &str) -> Declared {
    let mut out = Declared::default();
    // Byte offsets of every declaration line, so a mate's kind is read before the next one.
    let mut starts: Vec<usize> = Vec::new();
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        let t = line.trim_start();
        if t.starts_with("use ") || t.starts_with("connector ") || t.starts_with("mate ") {
            starts.push(at);
        }
        at += line.len();
    }
    for (i, &s) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(text.len());
        let block = &text[s..end];
        let t = block.trim_start();
        if let Some(rest) = t.strip_prefix("use part ") {
            let head = rest.lines().next().unwrap_or_default();
            if let Some(name) = head.rsplit_once(" as ").and_then(|(_, n)| name_after(n)) {
                out.parts.insert(name);
            }
        } else if let Some(rest) = t.strip_prefix("connector ") {
            if let Some(name) = name_after(rest) {
                out.connectors.insert(name);
            }
        } else if let Some(rest) = t.strip_prefix("mate ") {
            if let Some(name) = name_after(rest) {
                let after_name = &rest[rest.find(&name).map_or(0, |k| k + name.len())..];
                let kind = after_name
                    .find("mate =")
                    .map(|k| name_after(&after_name[k + "mate =".len()..]).unwrap_or_default())
                    .unwrap_or_default();
                out.mates.push((name, kind));
            }
        }
    }
    out
}

/// `Err` when the edit from `before` to `after` of the assembly at `path` places parts by hand.
pub fn refuse_hand_placement(path: &str, before: &str, after: &str) -> Result<(), ToolError> {
    let (was, now) = (declared(before), declared(after));
    if was.parts.is_empty() {
        return Ok(());
    }
    let new_parts: Vec<&String> = now.parts.difference(&was.parts).collect();
    let old_mates: BTreeSet<&str> = was.mates.iter().map(|(n, _)| n.as_str()).collect();
    let added_mates: Vec<&(String, String)> = now
        .mates
        .iter()
        .filter(|(n, _)| !old_mates.contains(n.as_str()))
        .collect();
    let seated_mates: Vec<&String> = added_mates
        .iter()
        .filter(|(_, k)| SEATED_KINDS.contains(&k.as_str()))
        .map(|(n, _)| n)
        .collect();
    // Connectors come with the mates that join them: those for other kinds of mate are allowed.
    let other_mates = added_mates.len() > seated_mates.len();
    let new_connectors: Vec<&String> = if other_mates {
        Vec::new()
    } else {
        now.connectors.difference(&was.connectors).collect()
    };
    if new_parts.is_empty() && seated_mates.is_empty() && new_connectors.is_empty() {
        return Ok(());
    }
    let list = |what: &str, names: &[&String]| {
        (!names.is_empty()).then(|| {
            format!(
                "{what} {}",
                names
                    .iter()
                    .map(|n| format!("`{n}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    };
    let adds: Vec<String> = [
        list("part occurrence", &new_parts),
        list("connector", &new_connectors),
        list("mate", &seated_mates),
    ]
    .into_iter()
    .flatten()
    .collect();
    let component = new_parts
        .first()
        .map(|n| n.as_str())
        .unwrap_or("clamp_front_left");
    let target = was
        .parts
        .iter()
        .next()
        .map(String::as_str)
        .unwrap_or("body");
    Err(ToolError::new(
        "use_seat_part",
        format!(
            "this edit adds {} to {path} by hand; place parts with design_seat_part (design.seat_part) instead, one call per component: it inserts the part, writes its connectors and mate, solves, and reports interference. For example: {{\"path\": \"{path}\", \"component\": \"{component}\", \"part\": \"/mechanical/skid_clamp.part\", \"target\": {{\"component\": \"{target}\", \"from\": [-30, 34, -100], \"direction\": [0, 0, 1]}}, \"own\": {{\"side\": \"+z\"}}, \"along\": [1, 0, 0]}}. A tube goes in a clamp's bore with \"target\": {{\"component\": \"<clamp>\", \"face\": \"<bore face>\"}} and \"own\": {{\"face\": \"<tube's outer face>\"}}. design_propose_edit still changes values on the assembly's existing occurrences, connectors and mates.",
            adds.join(", ")
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "assembly drone {\n    use part body as body\n    use part arm from \"./arms\" as arm {\n        fixed = false\n    }\n    connector arm_on = { occurrence = [{ name = body }], at = [0 mm, 0 mm, 0 mm] }\n    connector arm_at = { occurrence = [{ name = arm }], at = [0 mm, 0 mm, 0 mm] }\n    mate arm_fixed = { kind = { mate = fasten }, a = arm_on, b = arm_at }\n}\n";

    fn with(extra: &str) -> String {
        BASE.replace("\n}\n", &format!("\n{extra}}}\n"))
    }

    #[test]
    fn adding_a_part_connector_or_seated_mate_by_hand_is_refused_with_an_example_call() {
        let e = refuse_hand_placement(
            "/d.assembly",
            BASE,
            &with("    use part clamp as clamp_a {\n        fixed = false\n    }\n"),
        )
        .unwrap_err();
        assert_eq!(e.code, "use_seat_part");
        assert!(e.message.starts_with("this edit adds part occurrence `clamp_a` to /d.assembly by hand; place parts with design_seat_part"), "{}", e.message);
        assert!(
            e.message.contains("\"component\": \"clamp_a\"")
                && e.message.contains("\"target\": {\"component\": \"arm\""),
            "{}",
            e.message
        );
        let e = refuse_hand_placement("/d.assembly", BASE, &with("    connector c2 = { occurrence = [{ name = arm }] }\n    mate m2 = {\n        kind = { mate = planar },\n        a = arm_on, b = c2\n    }\n")).unwrap_err();
        assert!(
            e.message.contains("adds connector `c2`, mate `m2` to"),
            "{}",
            e.message
        );
    }

    #[test]
    fn values_removals_subassemblies_and_other_mates_stay_allowed() {
        // A changed value on an existing mate and a moved occurrence.
        let changed = BASE
            .replace(
                "at = [0 mm, 0 mm, 0 mm] }\n    mate",
                "at = [0 mm, 0 mm, 5 mm] }\n    mate",
            )
            .replace("mate = fasten", "mate = planar");
        assert!(refuse_hand_placement("/d.assembly", BASE, &changed).is_ok());
        // A removed mate.
        assert!(refuse_hand_placement(
            "/d.assembly",
            BASE,
            &BASE.replace(
                "    mate arm_fixed = { kind = { mate = fasten }, a = arm_on, b = arm_at }\n",
                ""
            )
        )
        .is_ok());
        // A subassembly, and a hinge with its connectors.
        assert!(refuse_hand_placement(
            "/d.assembly",
            BASE,
            &with("    use assembly gimbal as gimbal\n")
        )
        .is_ok());
        let hinge = with("    connector h1 = { occurrence = [{ name = arm }] }\n    connector h2 = { occurrence = [{ name = body }] }\n    mate hinge = { kind = { mate = revolute }, a = h1, b = h2 }\n");
        assert!(refuse_hand_placement("/d.assembly", BASE, &hinge).is_ok());
        // The first parts of an assembly that places none.
        let empty = "assembly skid {\n}\n";
        assert!(refuse_hand_placement(
            "/s.assembly",
            empty,
            "assembly skid {\n    use part tube as left\n    use part tube as right\n}\n"
        )
        .is_ok());
    }
}
