//! The argument registry: every operation a design file calls (feature kinds, their layouts and
//! forms, sketch entities and sketch constraints) with each argument's name, quantity kind,
//! default, allowed values and a one-line summary.
//!
//! It is the one source for:
//! - the quantity kind regeneration evaluates each numeric argument as (`Run::arg`);
//! - the language server's hovers, argument completion and signature help;
//! - the parameter tables in `docs/design/mcad/part/features.md` and `sketches.md`
//!   (`cargo test -p runmat-cad --test argument_docs`, with `RUNMAT_ARGUMENT_DOCS_WRITE=1` to
//!   rewrite them).
//!
//! Argument names are the written names: the model's serde field names, or the shorthand
//! arguments [`super::forms`] rewrites (`distance` and `through_all` for an extrude's extent). A
//! field nested in a record or in a choice's call is listed under its choice
//! (`form = counterbore(diameter, depth)`) or with a dotted name (`thin.thickness`).

mod features;
mod families;
mod sketch;

use crate::units::Dimension;

/// What an argument holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantity {
    Length,
    Angle,
    /// A plain number (a ratio, a fraction along an edge).
    Number,
    /// A whole number of copies or items.
    Count,
    /// Written alone (`through_all`) or `= true`.
    Flag,
    /// One of the listed words.
    Choice,
    /// Faces, edges, vertices or bodies, by reference path.
    Reference,
    /// A list of references.
    References,
    /// A sketch profile (`sketch.<name>`, a region of it or a chain).
    Profile,
    Plane,
    Axis,
    Point,
    /// Features named in a list.
    Features,
    Text,
    /// A record `{ … }` or a list of records.
    Record,
}

impl Quantity {
    /// The physical dimension a value of this kind evaluates to, for numeric kinds.
    pub fn dimension(self) -> Option<Dimension> {
        match self {
            Quantity::Length => Some(Dimension::LENGTH),
            Quantity::Angle => Some(Dimension::ANGLE),
            Quantity::Number | Quantity::Count => Some(Dimension::DIMENSIONLESS),
            _ => None,
        }
    }

    /// How a reader names the kind (`length`, `references`).
    pub fn label(self) -> &'static str {
        match self {
            Quantity::Length => "length",
            Quantity::Angle => "angle",
            Quantity::Number => "number",
            Quantity::Count => "count",
            Quantity::Flag => "flag",
            Quantity::Choice => "choice",
            Quantity::Reference => "reference",
            Quantity::References => "references",
            Quantity::Profile => "profile",
            Quantity::Plane => "plane",
            Quantity::Axis => "axis",
            Quantity::Point => "point",
            Quantity::Features => "features",
            Quantity::Text => "text",
            Quantity::Record => "record",
        }
    }

    /// The unit a value is usually written in.
    pub fn unit(self) -> Option<&'static str> {
        match self {
            Quantity::Length => Some("mm"),
            Quantity::Angle => Some("deg"),
            _ => None,
        }
    }
}

/// One allowed value. A choice with fields is written as a call (`counterbore(diameter = 9 mm,
/// depth = 5 mm)`).
#[derive(Clone, Copy, Debug)]
pub struct Allowed {
    pub word: &'static str,
    pub summary: &'static str,
    pub fields: &'static [Argument],
}

#[derive(Clone, Copy, Debug)]
pub struct Argument {
    pub name: &'static str,
    pub quantity: Quantity,
    /// The value used when the argument is left out (`None`: required, or absent).
    pub default: Option<&'static str>,
    pub required: bool,
    pub choices: &'static [Allowed],
    /// Other parameter paths regeneration reports for this argument (`second.distance`).
    pub sites: &'static [&'static str],
    pub summary: &'static str,
}

/// An optional argument.
pub(crate) const fn opt(name: &'static str, quantity: Quantity, summary: &'static str) -> Argument {
    Argument {
        name,
        quantity,
        default: None,
        required: false,
        choices: &[],
        sites: &[],
        summary,
    }
}

/// A required argument.
pub(crate) const fn req(name: &'static str, quantity: Quantity, summary: &'static str) -> Argument {
    Argument {
        required: true,
        ..opt(name, quantity, summary)
    }
}

impl Argument {
    pub(crate) const fn default(self, value: &'static str) -> Self {
        Argument {
            default: Some(value),
            ..self
        }
    }
    pub(crate) const fn choices(self, choices: &'static [Allowed]) -> Self {
        Argument { choices, ..self }
    }
    pub(crate) const fn sites(self, sites: &'static [&'static str]) -> Self {
        Argument { sites, ..self }
    }

    /// Nested fields (`thin.thickness`) are documented under their record, not offered as
    /// arguments of the call.
    pub fn is_nested(&self) -> bool {
        self.name.contains('.')
    }

    pub fn choice(&self, word: &str) -> Option<&'static Allowed> {
        self.choices.iter().find(|c| c.word == word)
    }
}

pub(crate) const fn choice(word: &'static str, summary: &'static str) -> Allowed {
    Allowed {
        word,
        summary,
        fields: &[],
    }
}

pub(crate) const fn call(
    word: &'static str,
    summary: &'static str,
    fields: &'static [Argument],
) -> Allowed {
    Allowed {
        word,
        summary,
        fields,
    }
}

/// Where an operation is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// `feature <name> = <word>(…)`.
    Feature,
    /// A sketch entity: `<name> = <word>(…)` inside `sketch … { }`.
    SketchEntity,
    /// A sketch constraint or dimension inside `sketch … { }`.
    SketchConstraint,
    /// A reference function inside a reference (`edges_between(a, b)`).
    Reference,
}

#[derive(Clone, Copy, Debug)]
pub struct Operation {
    /// The written word: `extrude`, or `<kind>.<variant>` (`plane.offset`).
    pub word: &'static str,
    pub group: Group,
    pub summary: &'static str,
    pub arguments: &'static [Argument],
}

impl Operation {
    pub fn argument(&self, name: &str) -> Option<&'static Argument> {
        self.arguments.iter().find(|a| a.name == name)
    }

    /// The arguments a call writes directly, in order.
    pub fn call_arguments(&self) -> impl Iterator<Item = &'static Argument> {
        self.arguments.iter().filter(|a| !a.is_nested())
    }

    /// `extrude(profile, distance, …)`.
    pub fn signature(&self) -> String {
        let names: Vec<&str> = self.call_arguments().map(|a| a.name).collect();
        format!("{}({})", self.word, names.join(", "))
    }
}

/// Every operation, features first.
pub fn operations() -> impl Iterator<Item = &'static Operation> {
    features::OPERATIONS
        .iter()
        .chain(families::OPERATIONS.iter())
        .chain(sketch::ENTITIES.iter())
        .chain(sketch::CONSTRAINTS.iter())
        .chain(sketch::REFERENCES.iter())
}

/// The operation written `word` (`extrude`, `plane.offset`, `circle`).
pub fn operation(word: &str) -> Option<&'static Operation> {
    operations().find(|o| o.word == word)
}

/// The operation written `word` in a group (a sketch's `distance` dimension, not a feature's
/// argument of that name).
pub fn operation_in(group: Group, word: &str) -> Option<&'static Operation> {
    operations().find(|o| o.group == group && o.word == word)
}

/// A choice written as a call (`counterbore`, `linear`) under one of `operation`'s arguments.
pub fn choice_call(
    operation: &Operation,
    word: &str,
) -> Option<(&'static Argument, &'static Allowed)> {
    operation.arguments.iter().find_map(|a| {
        a.choice(word)
            .filter(|c| !c.fields.is_empty())
            .map(|c| (a, c))
    })
}

/// The argument a regeneration parameter path names in a feature of kind `kind` (its
/// [`crate::model::visit::Params::kind_name`]): `distance`, `second.distance`, `form.depth`,
/// `variable[2].radius`. Index brackets are ignored.
pub fn parameter(kind: &str, path: &str) -> Option<&'static Argument> {
    let op = operation_in(Group::Feature, kind).or_else(|| {
        kind.split_once('.')
            .and_then(|(head, _)| operation_in(Group::Feature, head))
    })?;
    let path = strip_indices(path);
    if let Some(a) = op
        .arguments
        .iter()
        .find(|a| a.name == path || a.sites.contains(&path.as_str()))
    {
        return Some(a);
    }
    if let Some((head, field)) = path.split_once('.') {
        if let Some(a) = op.argument(head) {
            if let Some(f) = a
                .choices
                .iter()
                .flat_map(|c| c.fields)
                .find(|f| f.name == field)
            {
                return Some(f);
            }
        }
    }
    op.arguments
        .iter()
        .flat_map(|a| a.choices.iter().flat_map(|c| c.fields))
        .find(|f| f.name == path || f.sites.contains(&path.as_str()))
}

/// The dimension regeneration evaluates a feature parameter as.
pub fn dimension(kind: &str, path: &str) -> Option<Dimension> {
    parameter(kind, path).and_then(|a| a.quantity.dimension())
}

fn strip_indices(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut depth = 0;
    for c in path.chars() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// A Markdown table of an operation's arguments, for the docs and hovers.
pub fn markdown_table(op: &Operation) -> String {
    let mut out = String::from("| Argument | Kind | Default | Meaning |\n|---|---|---|---|\n");
    for a in op.arguments {
        let kind = match (a.quantity.unit(), a.choices.is_empty()) {
            (_, false) => a
                .choices
                .iter()
                .map(|c| format!("`{}`", c.word))
                .collect::<Vec<_>>()
                .join(" \\| "),
            (Some(unit), _) => format!("{} ({unit})", a.quantity.label()),
            (None, _) => a.quantity.label().to_string(),
        };
        let default = match (a.default, a.required) {
            (Some(d), _) => format!("`{d}`"),
            (None, true) => "required".into(),
            (None, false) => "".into(),
        };
        out.push_str(&format!(
            "| `{}` | {kind} | {default} | {} |\n",
            a.name, a.summary
        ));
    }
    out
}

#[cfg(test)]
mod tests;

/// A generated docs block for a marker line `<!-- arguments:<form> <words…> -->`:
/// - `operations <word>…`: a table of operations with their summaries and call arguments;
/// - `table <word>`: one operation's full argument table;
/// - `choices <word> <argument>`: an argument's call-form choices with their fields.
pub fn docs_block(spec: &str) -> Option<String> {
    let mut words = spec.split_whitespace();
    match words.next()? {
        "operations" => {
            let mut out = String::from("| Operation | Does | Arguments |\n|---|---|---|\n");
            for word in words {
                let op = operation_in(Group::Feature, word).or_else(|| operation(word))?;
                let args: Vec<String> = op
                    .call_arguments()
                    .map(|a| format!("`{}`", a.name))
                    .collect();
                out.push_str(&format!(
                    "| `{}` | {} | {} |\n",
                    op.word,
                    op.summary,
                    args.join(", ")
                ));
            }
            Some(out)
        }
        "table" => Some(markdown_table(operation(words.next()?)?)),
        "choices" => {
            let op = operation(words.next()?)?;
            let argument = op.argument(words.next()?)?;
            let mut out = format!(
                "| `{}` | Arguments | Gives |\n|---|---|---|\n",
                argument.name
            );
            for c in argument.choices {
                let fields: Vec<String> =
                    c.fields.iter().map(|f| format!("`{}`", f.name)).collect();
                out.push_str(&format!(
                    "| `{}` | {} | {} |\n",
                    c.word,
                    fields.join(", "),
                    c.summary
                ));
            }
            Some(out)
        }
        _ => None,
    }
}

/// Rewrites every `<!-- arguments:… -->` … `<!-- /arguments -->` block in a docs page from the
/// registry. Unknown markers are left as they are.
pub fn rewrite_docs(page: &str) -> String {
    let mut out = String::with_capacity(page.len());
    let mut rest = page;
    while let Some(start) = rest.find("<!-- arguments:") {
        let Some(head_end) = rest[start..].find("-->").map(|i| start + i + 3) else {
            break;
        };
        let Some(close) = rest[head_end..]
            .find("<!-- /arguments -->")
            .map(|i| head_end + i)
        else {
            break;
        };
        let spec = rest[start + "<!-- arguments:".len()..head_end - 3].trim();
        out.push_str(&rest[..head_end]);
        match docs_block(spec) {
            Some(block) => {
                out.push('\n');
                out.push_str(&block);
            }
            None => out.push_str(&rest[head_end..close]),
        }
        rest = &rest[close..];
        out.push_str("<!-- /arguments -->");
        rest = &rest["<!-- /arguments -->".len()..];
    }
    out.push_str(rest);
    out
}
