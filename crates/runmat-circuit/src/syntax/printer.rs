//! The `.circuit` printer: a [`Design`] back to source text that reads back bit for bit.

use super::*;

/// Why a design could not be printed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrintError(pub String);

impl std::fmt::Display for PrintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PrintError {}

pub(super) type Print = Result<String, PrintError>;

pub(super) fn name(text: &str) -> String {
    if is_bare_word(text) {
        text.to_string()
    } else {
        quote(text)
    }
}

pub(crate) fn endpoint(endpoint: &Endpoint) -> Print {
    let text = endpoint_text(endpoint);
    if endpoint_from_text(&text).as_ref() != Some(endpoint) {
        return Err(PrintError(format!(
            "the endpoint `{text}` has no text form"
        )));
    }
    let segments: Vec<&str> = text.split('.').collect();
    let reference = segments.iter().all(|segment| is_bare_key(segment))
        && endpoint_from_text(&segments.join(".")).as_ref() == Some(endpoint)
        && !text.contains('#');
    Ok(if reference { text } else { quote(&text) })
}

pub(super) fn si(quantity: &Quantity) -> Print {
    if !quantity.value.is_finite() {
        return Err(PrintError("a non-finite quantity".into()));
    }
    let number = format_f64(quantity.value);
    if quantity.dim == Dim::NONE {
        return Ok(number);
    }
    let symbol = SI_SYMBOLS
        .iter()
        .find(|(_, dim)| *dim == quantity.dim)
        .map(|(symbol, _)| *symbol)
        .ok_or_else(|| PrintError(format!("no unit to print the dimension {:?}", quantity.dim)))?;
    Ok(format!("{number} {symbol}"))
}

/// A percentage that reads back (divided by 100) as exactly `fraction`.
pub(super) fn percent(fraction: f64) -> Print {
    let first = format_f64(fraction * 100.0);
    if first.parse::<f64>().ok().map(|p| p / 100.0) == Some(fraction) {
        return Ok(format!("{first} %"));
    }
    for digits in 1..=20 {
        let text = format!("{:.*}", digits, fraction * 100.0);
        if text.parse::<f64>().ok().map(|p| p / 100.0) == Some(fraction) {
            return Ok(format!("{text} %"));
        }
    }
    Err(PrintError(format!(
        "the relative tolerance {fraction} has no exact percentage"
    )))
}

/// A param as written: the `.m` call that computes it, else its value.
pub(crate) fn param_written(param: &crate::ir::Param) -> Print {
    match (&param.call, &param.written) {
        (Some(call), _) => Ok(call.text.clone()),
        (None, Some(written)) => Ok(written.clone()),
        (None, None) => param_text(&param.value),
    }
}

pub(super) fn param_text(value: &ParamValue) -> Print {
    Ok(match value {
        ParamValue::Text(text) => quote(text),
        ParamValue::Bool(flag) => flag.to_string(),
        ParamValue::Quantity { nominal, tol } => {
            let nominal_text = si(nominal)?;
            let with_unit = |v: f64| si(&Quantity::new(v, nominal.dim));
            match tol {
                Tolerance::Exact => nominal_text,
                Tolerance::Relative { minus, plus } if minus == plus => {
                    format!("{nominal_text} ± {}", percent(*plus)?)
                }
                Tolerance::Relative { minus, plus } => {
                    format!(
                        "{nominal_text} ± [{}, {}]",
                        percent(*plus)?,
                        percent(*minus)?
                    )
                }
                Tolerance::Absolute { minus, plus } if minus == plus => {
                    format!("{nominal_text} ± {}", with_unit(*plus)?)
                }
                Tolerance::Absolute { minus, plus } => {
                    format!(
                        "{nominal_text} ± [{}, {}]",
                        with_unit(*plus)?,
                        with_unit(*minus)?
                    )
                }
                Tolerance::Range { min, max } => format!(
                    "{{ nominal = {nominal_text}, range = [{}, {}] }}",
                    with_unit(*min)?,
                    with_unit(*max)?
                ),
            }
        }
    })
}

pub(super) fn key(text: &str) -> String {
    if is_bare_key(text) {
        text.to_string()
    } else {
        quote(text)
    }
}

pub(super) fn record(entries: &[(String, String)]) -> String {
    if entries.is_empty() {
        return "{}".into();
    }
    let inline = format!(
        "{{ {} }}",
        entries
            .iter()
            .map(|(k, v)| format!("{} = {v}", key(k)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if inline.len() <= 84 && !inline.contains('\n') {
        return inline;
    }
    let mut out = String::from("{\n");
    for (index, (k, v)) in entries.iter().enumerate() {
        out.push_str(INDENT);
        out.push_str(&format!(
            "{} = {}",
            key(k),
            v.replace('\n', &format!("\n{INDENT}"))
        ));
        if index + 1 < entries.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push('}');
    out
}

pub(super) fn list(items: &[String]) -> String {
    format!("[{}]", items.join(", "))
}

pub(super) fn names(items: &[String]) -> String {
    list(&items.iter().map(|item| name(item)).collect::<Vec<_>>())
}

pub(super) fn params_entries(
    params: &BTreeMap<String, Param>,
    reserved: &[&str],
) -> Result<Vec<(String, String)>, PrintError> {
    params
        .iter()
        .map(|(k, p)| {
            if reserved.contains(&k.as_str()) {
                return Err(PrintError(format!(
                    "a param named `{k}` collides with a field"
                )));
            }
            Ok((k.clone(), param_written(p)?))
        })
        .collect()
}

pub(super) fn part_entries(part: &PartRef) -> Vec<(String, String)> {
    match part {
        PartRef::Part(part) => vec![("part".into(), name(part))],
        PartRef::Entity { entity, package } => {
            let mut out = vec![("entity".into(), name(entity))];
            if let Some(package) = package {
                out.push(("package".into(), name(package)));
            }
            out
        }
    }
}

pub(super) const COMPONENT_FIELDS: &[&str] = &[
    "part",
    "entity",
    "package",
    "classes",
    "power",
    "renamed_from",
    "variants",
    "properties",
];

pub(super) fn fitting_text(fitting: &Fitting) -> Print {
    Ok(match fitting {
        Fitting::Fitted => "fitted".into(),
        Fitting::NotFitted => "not_fitted".into(),
        Fitting::Alternate { part, params } => {
            let mut entries = part.as_ref().map(part_entries).unwrap_or_default();
            entries.extend(params_entries(params, &["part", "entity", "package"])?);
            if entries.is_empty() {
                return Err(PrintError(
                    "an empty alternate fitting has no text form".into(),
                ));
            }
            record(&entries)
        }
    })
}

pub(super) fn fittings(map: &BTreeMap<String, Fitting>) -> Print {
    let entries: Result<Vec<_>, _> = map
        .iter()
        .map(|(k, f)| Ok((k.clone(), fitting_text(f)?)))
        .collect();
    Ok(record(&entries?))
}

/// Prints a design as `.circuit` source (sheets in name order, declarations in model order).
pub fn print_circuit(design: &Design) -> Print {
    let mut body: Vec<String> = Vec::new();
    if design.root != "main" {
        body.push(format!("root = {}", name(&design.root)));
    }
    for (k, param) in &design.params {
        body.push(format!("param {} = {}", name(k), param_written(param)?));
    }
    for (k, class) in &design.net_classes {
        if &class.name != k {
            return Err(PrintError(format!(
                "net class `{k}` is keyed by another name"
            )));
        }
        body.push(format!(
            "net_class {} = {}",
            name(k),
            record(&params_entries(&class.params, &[])?)
        ));
    }
    for (k, interface) in &design.interfaces {
        let mut entries = vec![("members".to_string(), names(&interface.members))];
        if !interface.diff_pairs.is_empty() {
            let pairs: Vec<String> = interface
                .diff_pairs
                .iter()
                .map(|(p, n)| names(&[p.clone(), n.clone()]))
                .collect();
            entries.push(("diff_pairs".into(), list(&pairs)));
        }
        body.push(format!("interface {} = {}", name(k), record(&entries)));
    }
    for (k, variant) in &design.variants {
        let mut entries = Vec::new();
        if let Some(base) = &variant.base {
            entries.push(("base".to_string(), name(base)));
        }
        if !variant.overrides.is_empty() {
            entries.push(("overrides".into(), fittings(&variant.overrides)?));
        }
        body.push(format!("variant {} = {}", name(k), record(&entries)));
    }
    for (k, part) in &design.parts {
        if &part.mpn != k {
            return Err(PrintError(format!("part `{k}` is keyed by another MPN")));
        }
        body.push(super::part_decl::statement(part)?);
    }
    for (k, sheet) in &design.sheets {
        if &sheet.name != k {
            return Err(PrintError(format!("sheet `{k}` is keyed by another name")));
        }
        if !body.is_empty() {
            body.push(String::new());
        }
        body.push(print_sheet(sheet)?);
    }
    for sim in design.simulations.values() {
        if !body.is_empty() {
            body.push(String::new());
        }
        body.push(simulation_print::print(sim));
    }
    let mut out = String::from("runmat 0.7\nstd 1\n\n");
    for u in &design.uses {
        out.push_str(&use_text(u));
        out.push('\n');
    }
    if !design.uses.is_empty() {
        out.push('\n');
    }
    out.push_str(&format!("circuit {} {{\n", name(&design.name)));
    for statement in body {
        if statement.is_empty() {
            out.push('\n');
            continue;
        }
        out.push_str(INDENT);
        out.push_str(&statement.replace('\n', &format!("\n{INDENT}")));
        out.push('\n');
    }
    out.push_str("}\n");
    Ok(out)
}

/// One `component` statement, as the sheet prints it.
pub(crate) fn component_statement(component: &ComponentDecl) -> Print {
    let mut entries = part_entries(&component.part);
    if !component.classes.is_empty() {
        entries.push(("classes".into(), names(&component.classes)));
    }
    if let Some(power) = &component.power {
        entries.push(("power".into(), si(power)?));
    }
    if let Some(renamed) = &component.renamed_from {
        entries.push(("renamed_from".into(), name(renamed)));
    }
    if !component.variants.is_empty() {
        entries.push(("variants".into(), fittings(&component.variants)?));
    }
    if !component.properties.is_empty() {
        let properties: Vec<(String, String)> = component
            .properties
            .iter()
            .map(|(k, v)| (k.clone(), quote(v)))
            .collect();
        entries.push(("properties".into(), record(&properties)));
    }
    entries.extend(params_entries(&component.params, COMPONENT_FIELDS)?);
    Ok(format!(
        "component {} = {}",
        name(&component.name),
        record(&entries)
    ))
}

/// One `connect = [...]` statement.
pub(crate) fn connection_statement(connection: &Connection) -> Print {
    let ends: Result<Vec<String>, _> = connection.ends.iter().map(endpoint).collect();
    Ok(format!("connect = {}", list(&ends?)))
}

/// One `port` statement.
pub(crate) fn port_statement(port: &PortDecl) -> Print {
    let dir = DIRECTIONS
        .iter()
        .find(|(_, d)| *d == port.dir)
        .map(|(w, _)| *w)
        .unwrap_or("passive");
    let kind = match &port.kind {
        PortKind::Signal => String::new(),
        PortKind::Bus(members) => format!(" = bus {}", names(members)),
        PortKind::Interface(interface) => format!(" = interface {}", name(interface)),
    };
    Ok(format!("port {} : {dir}{kind}", name(&port.name)))
}

/// One `net` statement.
pub(crate) fn net_statement(net: &NetDecl) -> Print {
    let mut entries = Vec::new();
    if let Some(class) = &net.class {
        entries.push(("class".to_string(), name(class)));
    }
    if net.global {
        entries.push(("global".into(), "true".into()));
    }
    if let Some(renamed) = &net.renamed_from {
        entries.push(("renamed_from".into(), name(renamed)));
    }
    entries.extend(params_entries(
        &net.params,
        &["class", "global", "renamed_from"],
    )?);
    Ok(if entries.is_empty() {
        format!("net {}", name(&net.name))
    } else {
        format!("net {} = {}", name(&net.name), record(&entries))
    })
}

/// One `bus` statement.
pub(crate) fn bus_statement(bus: &BusDecl) -> Print {
    Ok(format!("bus {} = {}", name(&bus.name), names(&bus.members)))
}

/// One `harness` statement.
pub(crate) fn harness_statement(harness: &HarnessDecl) -> Print {
    Ok(format!(
        "harness {} = {}",
        name(&harness.name),
        name(&harness.interface)
    ))
}

/// One `instance` statement.
pub(crate) fn instance_statement(instance: &SheetInstance) -> Print {
    let mut entries = vec![("sheet".to_string(), name(&instance.sheet))];
    if let Some(repeat) = instance.repeat {
        entries.push(("repeat".into(), repeat.to_string()));
    }
    if let Some(template) = &instance.designators {
        entries.push(("designators".into(), quote(template)));
    }
    if let Some(renamed) = &instance.renamed_from {
        entries.push(("renamed_from".into(), name(renamed)));
    }
    if !instance.bindings.is_empty() {
        let mut bind = Vec::new();
        for (port, binding) in &instance.bindings {
            if bind.iter().any(|(k, _): &(String, String)| k == port) {
                return Err(PrintError(format!(
                    "instance `{}` binds `{port}` twice",
                    instance.name
                )));
            }
            let text = match binding {
                Binding::Shared(end) => endpoint(end)?,
                Binding::BusIndexed { bus, offset } => {
                    format!("{{ bus = {}, offset = {offset} }}", name(bus))
                }
                Binding::NetIndexed { prefix, offset } => {
                    format!("{{ net_prefix = {}, offset = {offset} }}", name(prefix))
                }
            };
            bind.push((port.clone(), text));
        }
        entries.push(("bind".into(), record(&bind)));
    }
    Ok(format!(
        "instance {} = {}",
        name(&instance.name),
        record(&entries)
    ))
}

/// The `no_erc = [...]` statement of these endpoints.
pub(crate) fn no_erc_statement(ends: &[Endpoint]) -> Print {
    let ends: Result<Vec<String>, _> = ends.iter().map(endpoint).collect();
    Ok(format!("no_erc = {}", list(&ends?)))
}

/// One `text` statement.
pub(crate) fn text_statement(t: &crate::ir::TextDecl) -> String {
    format!("text {} = {}", name(&t.name), quote(&t.text))
}

/// One `diff_pair` statement.
pub(crate) fn diff_pair_statement(pair: &DiffPairDecl) -> Print {
    let mut entries = vec![
        ("p".to_string(), name(&pair.p)),
        ("n".to_string(), name(&pair.n)),
    ];
    if let Some(class) = &pair.class {
        entries.push(("class".into(), name(class)));
    }
    Ok(format!(
        "diff_pair {} = {}",
        name(&pair.name),
        record(&entries)
    ))
}

/// One `blanket` statement.
pub(crate) fn blanket_statement(blanket: &BlanketDecl) -> Print {
    let mut entries = vec![("nets".to_string(), names(&blanket.nets))];
    if let Some(class) = &blanket.class {
        entries.push(("class".into(), name(class)));
    }
    entries.extend(params_entries(&blanket.params, &["nets", "class", "global", "renamed_from"])?);
    Ok(format!("blanket {} = {}", name(&blanket.name), record(&entries)))
}

pub(super) fn print_sheet(sheet: &SheetDef) -> Print {
    let mut lines = Vec::new();
    for (k, param) in &sheet.params {
        lines.push(format!("param {} = {}", name(k), param_written(param)?));
    }
    for port in &sheet.ports {
        lines.push(port_statement(port)?);
    }
    for component in &sheet.components {
        lines.push(component_statement(component)?);
    }
    for net in &sheet.nets {
        lines.push(net_statement(net)?);
    }
    for bus in &sheet.buses {
        lines.push(bus_statement(bus)?);
    }
    for harness in &sheet.harnesses {
        lines.push(harness_statement(harness)?);
    }
    for instance in &sheet.instances {
        lines.push(instance_statement(instance)?);
    }
    for connection in &sheet.connections {
        lines.push(connection_statement(connection)?);
    }
    if !sheet.no_erc.is_empty() {
        lines.push(no_erc_statement(&sheet.no_erc)?);
    }
    for pair in &sheet.diff_pairs {
        lines.push(diff_pair_statement(pair)?);
    }
    for blanket in &sheet.blankets {
        lines.push(blanket_statement(blanket)?);
    }
    for t in &sheet.texts {
        lines.push(text_statement(t));
    }
    let mut out = format!("sheet {} {{\n", name(&sheet.name));
    for line in lines {
        out.push_str(INDENT);
        out.push_str(&line.replace('\n', &format!("\n{INDENT}")));
        out.push('\n');
    }
    out.push('}');
    Ok(out)
}
