//! The assembly text a seating writes: the inserted occurrence, `fixed = false` on the one it
//! moves, its two connectors and its mate, all named after the component so seating it again
//! replaces them, and (for mates that leave freedoms) its starting placement.

use super::Frame;

/// A connector frame as written: origin in mm, Z and X as unit vectors.
pub struct FrameText {
    pub at_mm: [f64; 3],
    pub z: [f64; 3],
    pub x: [f64; 3],
}

/// The target side of the mate: a connector the assembly already has, or a new one on the
/// target component.
pub enum TargetLine {
    Existing(String),
    New(FrameText),
}

pub fn is_name(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn own_connector(name: &str) -> String {
    format!("{name}_seat")
}
fn target_connector(name: &str) -> String {
    format!("{name}_seat_on")
}
pub fn mate_name(name: &str) -> String {
    format!("{name}_seated")
}
fn marker(name: &str) -> String {
    format!("// {name}: seated by design.seat_part")
}

/// `text` without the seat a previous seating of `name` wrote.
pub fn without_seat(text: &str, name: &str) -> String {
    let starts = [
        marker(name),
        format!("connector {} =", own_connector(name)),
        format!("connector {} =", target_connector(name)),
        format!("mate {} =", mate_name(name)),
    ];
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        if !starts.iter().any(|s| line.trim_start().starts_with(s.as_str())) {
            out.push_str(line);
        }
    }
    out
}

/// The line that declares occurrence `name` (`use part … as name`), by byte range.
fn occurrence_line(text: &str, name: &str) -> Option<(usize, usize)> {
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        let body = line.trim_end();
        let trimmed = body.trim_start();
        if trimmed.starts_with("use ") {
            let tail = trimmed.strip_suffix('{').map(str::trim_end).unwrap_or(trimmed);
            if tail.ends_with(&format!(" as {name}")) {
                return Some((at, at + line.len()));
            }
        }
        at += line.len();
    }
    None
}

pub fn declares_occurrence(text: &str, name: &str) -> bool {
    occurrence_line(text, name).is_some()
}

pub fn declares_connector(text: &str, connector: &str) -> bool {
    text.lines().any(|l| l.trim_start().starts_with(&format!("connector {connector} =")))
}

/// The name a part file declares (`part skid_clamp {`).
pub fn part_symbol(text: &str) -> Option<String> {
    text.lines().find_map(|l| {
        let rest = l.trim_start().strip_prefix("part ")?;
        let symbol: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
        (!symbol.is_empty() && rest[symbol.len()..].trim_start().starts_with('{')).then_some(symbol)
    })
}

fn folder(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some((d, _)) => d,
        None => "",
    }
}

/// `to` (a folder) relative to `from` (a folder), as `use … from` reads it: `./a/b`, `../c`.
fn relative(from: &str, to: &str) -> String {
    let a: Vec<&str> = from.split('/').filter(|s| !s.is_empty()).collect();
    let b: Vec<&str> = to.split('/').filter(|s| !s.is_empty()).collect();
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let ups = a.len() - common;
    let down = b[common..].join("/");
    match (ups, down.is_empty()) {
        (0, _) => format!("./{down}"),
        (n, true) => vec![".."; n].join("/"),
        (n, false) => format!("{}/{down}", vec![".."; n].join("/")),
    }
}

/// Byte offset of the assembly's closing brace (the last line that is `}`).
fn closing(text: &str) -> usize {
    text.rfind("\n}").map(|i| i + 1).unwrap_or(text.len())
}

/// `text` with a new occurrence `name` of the part `symbol` at `part` (a workspace path), free to
/// move, with `values` as its input values.
pub fn insert_occurrence(text: &str, assembly: &str, part: &str, symbol: &str, name: &str, values: &[(String, String)]) -> String {
    let (from, to) = (folder(assembly), folder(part));
    let source = if from == to { String::new() } else { format!(" from \"{}\"", relative(from, to)) };
    let mut block = format!("\n    {}\n    use part {symbol}{source} as {name} {{\n        fixed = false\n", marker(name));
    if !values.is_empty() {
        let list: Vec<String> = values.iter().map(|(k, v)| format!("{k} = {v}")).collect();
        block.push_str(&format!("        configuration = {{ select = inputs, values = {{ {} }} }}\n", list.join(", ")));
    }
    block.push_str("    }\n");
    let at = closing(text);
    format!("{}{block}{}", &text[..at], &text[at..])
}

/// `text` with occurrence `name` free to move (`fixed = false`), or `None` when the assembly
/// doesn't declare it at the top level.
pub fn free_occurrence(text: &str, name: &str) -> Option<String> {
    let (start, end) = occurrence_line(text, name)?;
    let line = &text[start..end];
    let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
    if !line.trim_end().ends_with('{') {
        let head = line.trim_end();
        return Some(format!("{}{head} {{\n{indent}    fixed = false\n{indent}}}\n{}", &text[..start], &text[end..]));
    }
    // A block: replace its `fixed` line, or add one first.
    let close = format!("{indent}}}");
    let mut at = end;
    for l in text[end..].split_inclusive('\n') {
        let t = l.trim_start();
        if l.trim_end() == close {
            break;
        }
        if t.starts_with("fixed =") {
            let new_line = format!("{indent}    fixed = false\n");
            return Some(format!("{}{new_line}{}", &text[..at], &text[at + l.len()..]));
        }
        at += l.len();
    }
    Some(format!("{}{indent}    fixed = false\n{}", &text[..end], &text[end..]))
}

fn number(v: f64) -> String {
    let s = format!("{v}");
    if s == "-0" {
        "0".into()
    } else {
        s
    }
}

fn mm(p: [f64; 3]) -> String {
    format!("[{} mm, {} mm, {} mm]", number(p[0]), number(p[1]), number(p[2]))
}

fn direction(v: [f64; 3]) -> String {
    format!("[{}, {}, {}]", number(v[0]), number(v[1]), number(v[2]))
}

fn occurrence_path(name: &str) -> String {
    let parts: Vec<String> = name.split('/').map(|n| format!("{{ name = {n} }}")).collect();
    format!("[{}]", parts.join(", "))
}

fn frame_fields(f: &FrameText) -> String {
    format!("at = {}, z = {}, x = {}", mm(f.at_mm), direction(f.z), direction(f.x))
}

/// The component's connector at a frame of its part; `flip` turns it face to face.
pub fn own_explicit(name: &str, frame: &Frame, flip: bool) -> String {
    let f = super::frame_text(frame);
    format!(
        "connector {} = {{ occurrence = {}, {}{} }}",
        own_connector(name),
        occurrence_path(name),
        frame_fields(&f),
        if flip { ", flip = true" } else { "" }
    )
}

/// The component's connector on a frame its part publishes.
pub fn own_published(name: &str, published: &str, flip: bool) -> String {
    format!(
        "connector {} = {{ occurrence = {}, anchor = {{ anchor = published, name = {published} }}{} }}",
        own_connector(name),
        occurrence_path(name),
        if flip { ", flip = true" } else { "" }
    )
}

/// `text` with the seat: the target's connector (when new), the component's, and the mate.
pub fn add_seat(text: &str, name: &str, target: &str, kind: &str, target_line: &TargetLine, own_line: &str) -> String {
    let a = match target_line {
        TargetLine::Existing(c) => c.clone(),
        TargetLine::New(_) => target_connector(name),
    };
    let mut block = String::new();
    if !text.contains(&marker(name)) {
        block.push_str(&format!("\n    {}\n", marker(name)));
    }
    if let TargetLine::New(f) = target_line {
        block.push_str(&format!("    connector {a} = {{ occurrence = {}, {} }}\n", occurrence_path(target), frame_fields(f)));
    }
    block.push_str(&format!("    {own_line}\n"));
    block.push_str(&format!("    mate {} = {{ kind = {{ mate = {kind} }}, a = {a}, b = {} }}\n", mate_name(name), own_connector(name)));
    let at = closing(text);
    format!("{}{block}{}", &text[..at], &text[at..])
}

/// `text` with occurrence `name`'s placement at `translation` (mm) turned `angle` degrees about
/// `axis`: its `placement` line replaced, or one added to its block.
pub fn set_placement(text: &str, name: &str, translation: [f64; 3], axis: [f64; 3], angle: f64) -> Option<String> {
    let (start, end) = occurrence_line(text, name)?;
    let line = &text[start..end];
    if !line.trim_end().ends_with('{') {
        return None;
    }
    let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
    let placement = format!(
        "{indent}    placement = {{ placement = offset, offset = {{ translation = {}, rotation = {{ rotation = axis_angle, axis = {}, angle = {} deg }} }} }}\n",
        mm(translation),
        direction(axis),
        number(angle)
    );
    let close = format!("{indent}}}");
    let mut at = end;
    for l in text[end..].split_inclusive('\n') {
        if l.trim_end() == close {
            break;
        }
        if l.trim_start().starts_with("placement =") {
            return Some(format!("{}{placement}{}", &text[..at], &text[at + l.len()..]));
        }
        at += l.len();
    }
    Some(format!("{}{placement}{}", &text[..end], &text[end..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASM: &str = "runmat 0.7\nstd 1\n\nassembly a {\n    use part plate as plate\n    use part block as block {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 5 mm] } }\n    }\n}\n";

    #[test]
    fn an_occurrence_frees_and_a_new_one_inserts_beside_its_folder() {
        let freed = free_occurrence(ASM, "block").unwrap();
        assert!(freed.contains("    use part block as block {\n        fixed = false\n        placement"), "{freed}");
        let freed = free_occurrence(ASM, "plate").unwrap();
        assert!(freed.contains("    use part plate as plate {\n        fixed = false\n    }\n"), "{freed}");
        assert!(free_occurrence(ASM, "none").is_none());
        let inserted = insert_occurrence(ASM, "/d/a.assembly", "/d/mech/clamp.part", "clamp", "clamp_fl", &[("bore".into(), "10.2 mm".into())]);
        assert!(inserted.contains("use part clamp from \"./mech\" as clamp_fl {\n        fixed = false\n        configuration = { select = inputs, values = { bore = 10.2 mm } }\n    }\n}\n"), "{inserted}");
        assert_eq!(relative("/d/x", "/d/y/z"), "../y/z");
        assert_eq!(relative("/d/x", "/d"), "..");
        assert_eq!(part_symbol("runmat 0.7\n\npart skid_clamp {\n}\n").as_deref(), Some("skid_clamp"));
    }

    #[test]
    fn a_seat_is_written_once_and_replaced_by_the_next() {
        let f = FrameText { at_mm: [1.0, 2.0, -0.0], z: [0.0, 0.0, -1.0], x: [1.0, 0.0, 0.0] };
        let own = "connector block_seat = { occurrence = [{ name = block }], at = [0 mm, 0 mm, 5 mm], z = [0, 0, 1], x = [1, 0, 0], flip = true }";
        let once = add_seat(ASM, "block", "plate", "fasten", &TargetLine::New(f), own);
        assert!(once.contains("connector block_seat_on = { occurrence = [{ name = plate }], at = [1 mm, 2 mm, 0 mm], z = [0, 0, -1], x = [1, 0, 0] }\n"), "{once}");
        assert!(once.contains("mate block_seated = { kind = { mate = fasten }, a = block_seat_on, b = block_seat }\n}\n"), "{once}");
        let cleared = without_seat(&once, "block");
        assert!(!cleared.contains("block_seat") && !cleared.contains("seated by"), "{cleared}");
        let placed = set_placement(&free_occurrence(ASM, "block").unwrap(), "block", [1.0, 2.0, 3.0], [0.0, 0.0, 1.0], 90.0).unwrap();
        assert!(placed.contains("        placement = { placement = offset, offset = { translation = [1 mm, 2 mm, 3 mm], rotation = { rotation = axis_angle, axis = [0, 0, 1], angle = 90 deg } } }\n    }"), "{placed}");
    }
}
