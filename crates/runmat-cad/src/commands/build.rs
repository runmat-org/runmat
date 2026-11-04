//! Small constructors so the catalog reads as a table of commands.

use super::*;

/// Starts a command. Defaults: enabled always, no inputs, a display-only edit, no preview.
pub fn cmd(id: &str, label: &str, icon: &str, editors: &[EditorKind]) -> CommandDescriptor {
    CommandDescriptor {
        id: id.to_owned(),
        label: label.to_owned(),
        description: String::new(),
        keywords: Vec::new(),
        editors: editors.to_vec(),
        icon: icon.to_owned(),
        inputs: Vec::new(),
        enabled: Predicate::Always,
        edit: edit(EditOp::View, Section::None),
        preview: false,
        context: Vec::new(),
        placements: Vec::new(),
    }
}

impl CommandDescriptor {
    pub fn desc(mut self, text: &str) -> Self {
        self.description = text.to_owned();
        self
    }
    pub fn kw(mut self, words: &[&str]) -> Self {
        self.keywords.extend(words.iter().map(|w| (*w).to_owned()));
        self
    }
    pub fn inputs(mut self, fields: Vec<Field>) -> Self {
        self.inputs = fields;
        self
    }
    pub fn when(mut self, predicate: Predicate) -> Self {
        self.enabled = predicate;
        self
    }
    pub fn edit(mut self, edit: ModelEdit) -> Self {
        self.edit = edit;
        self
    }
    /// Live preview while the inputs change.
    pub fn preview(mut self) -> Self {
        self.preview = true;
        self
    }
    pub fn on(mut self, kinds: &[SelectKind]) -> Self {
        self.context.extend_from_slice(kinds);
        self
    }
}

/* ----------------------------------------------------------------------------------------- */
/* Edits                                                                                     */
/* ----------------------------------------------------------------------------------------- */

pub fn edit(op: EditOp, section: Section) -> ModelEdit {
    ModelEdit {
        op,
        section,
        item: None,
        variant: None,
        format: None,
    }
}

impl ModelEdit {
    pub fn item(mut self, item: &str) -> Self {
        self.item = Some(item.to_owned());
        self
    }
    pub fn variant(mut self, tag: &str, value: &str) -> Self {
        self.variant = Some(Variant {
            tag: tag.to_owned(),
            value: value.to_owned(),
        });
        self
    }
    pub fn format(mut self, format: &str) -> Self {
        self.format = Some(format.to_owned());
        self
    }
}

/// Adds a feature of the model's `FeatureKind` tag.
pub fn add_feature(feature: &str) -> ModelEdit {
    edit(EditOp::Add, Section::Features).item(feature)
}

/// Adds a feature whose kind has a tagged sub-kind (`direct`/`move_faces`, `surface`/`kind`…).
pub fn add_feature_variant(feature: &str, tag: &str, value: &str) -> ModelEdit {
    add_feature(feature).variant(tag, value)
}

pub fn analyze(query: &str) -> ModelEdit {
    edit(EditOp::Analyze, Section::None).item(query)
}

pub fn check(check: &str) -> ModelEdit {
    edit(EditOp::Check, Section::None).item(check)
}

pub fn view_action(action: &str) -> ModelEdit {
    edit(EditOp::View, Section::None).item(action)
}

pub fn export(format: &str) -> ModelEdit {
    edit(EditOp::Export, Section::None).format(format)
}

/* ----------------------------------------------------------------------------------------- */
/* Fields                                                                                    */
/* ----------------------------------------------------------------------------------------- */

fn field(id: &str, label: &str, kind: FieldKind) -> Field {
    Field {
        id: id.to_owned(),
        label: label.to_owned(),
        kind,
        required: true,
        help: None,
        visible_when: None,
        manipulator: None,
    }
}

impl Field {
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }
    pub fn help(mut self, text: &str) -> Self {
        self.help = Some(text.to_owned());
        self
    }
    /// Shown only while `field` has one of `values`.
    pub fn shown(mut self, field: &str, values: &[&str]) -> Self {
        self.visible_when = Some(FieldCondition {
            field: field.to_owned(),
            any_of: values.iter().map(|v| (*v).to_owned()).collect(),
        });
        self
    }
    /// Shown only while the boolean `field` is checked.
    pub fn if_checked(self, field: &str) -> Self {
        self.shown(field, &["true"])
    }
    pub fn drag(mut self, manipulator: Manipulator) -> Self {
        self.manipulator = Some(manipulator);
        self
    }
    pub fn min(mut self, value: f64) -> Self {
        match &mut self.kind {
            FieldKind::Quantity { min, .. } => *min = Some(value),
            FieldKind::Count { min, .. } => *min = Some(value as i64),
            _ => {}
        }
        self
    }
    pub fn max(mut self, value: f64) -> Self {
        match &mut self.kind {
            FieldKind::Quantity { max, .. } => *max = Some(value),
            FieldKind::Count { max, .. } => *max = Some(value as i64),
            _ => {}
        }
        self
    }
}

pub fn quantity_spec(name: &str) -> QuantitySpec {
    let (dimension, unit) = match name {
        "length" => (Dimension::LENGTH, "mm"),
        "angle" => (Dimension::ANGLE, "deg"),
        "mass" => (Dimension::MASS, "kg"),
        "area" => (Dimension::AREA, "mm^2"),
        "ratio" => (Dimension::DIMENSIONLESS, ""),
        "scale" => (Dimension::DIMENSIONLESS, ""),
        "time" => (time(), "s"),
        "density" => (Dimension::MASS.divide(Dimension::LENGTH.power(3)), "kg/m^3"),
        "force" => (force(), "N"),
        "torque" => (force().multiply(Dimension::LENGTH), "N*m"),
        "speed" => (Dimension::LENGTH.divide(time()), "mm/s"),
        "angular_speed" => (Dimension::ANGLE.divide(time()), "rpm"),
        "stiffness" => (force().divide(Dimension::LENGTH), "N/mm"),
        "voltage" => (voltage(), "V"),
        "power" => (voltage().multiply(current()), "W"),
        "current" => (current(), "A"),
        "resistance" => (voltage().divide(current()), "ohm"),
        "frequency" => (Dimension::DIMENSIONLESS.divide(time()), "Hz"),
        "temperature" => (temperature(), "degC"),
        other => panic!("no quantity named {other}"),
    };
    QuantitySpec {
        name: name.to_owned(),
        dimension,
        unit: unit.to_owned(),
    }
}

fn base(index: usize) -> Dimension {
    let mut d = Dimension::DIMENSIONLESS;
    d.base[index] = 1;
    d
}
// SI base order: length, mass, time, current, temperature, amount, luminous intensity, (spare).
fn time() -> Dimension {
    base(2)
}
fn current() -> Dimension {
    base(3)
}
fn temperature() -> Dimension {
    base(4)
}
fn force() -> Dimension {
    Dimension::MASS
        .multiply(Dimension::LENGTH)
        .divide(time().power(2))
}
fn voltage() -> Dimension {
    force()
        .multiply(Dimension::LENGTH)
        .divide(time())
        .divide(current())
}

/// A physical value with a default, in the quantity's display unit.
pub fn qty(id: &str, label: &str, quantity: &str, default: Option<f64>) -> Field {
    let spec = quantity_spec(quantity);
    field(
        id,
        label,
        FieldKind::Quantity {
            quantity: spec.name,
            unit: spec.unit,
            default,
            min: None,
            max: None,
            step: None,
        },
    )
}

pub fn length(id: &str, label: &str, default: f64) -> Field {
    qty(id, label, "length", Some(default)).min(0.0)
}

/// A signed length (offsets that can go either way).
pub fn offset(id: &str, label: &str, default: f64) -> Field {
    qty(id, label, "length", Some(default))
}

pub fn angle(id: &str, label: &str, default: f64) -> Field {
    qty(id, label, "angle", Some(default))
}

pub fn count(id: &str, label: &str, default: i64, min: i64) -> Field {
    field(
        id,
        label,
        FieldKind::Count {
            default: Some(default),
            min: Some(min),
            max: None,
        },
    )
}

fn options(pairs: &[(&str, &str)]) -> Vec<ChoiceOption> {
    pairs
        .iter()
        .map(|(value, label)| ChoiceOption {
            value: (*value).to_owned(),
            label: (*label).to_owned(),
        })
        .collect()
}

/// A dropdown; the first option is the default.
pub fn choice(id: &str, label: &str, pairs: &[(&str, &str)]) -> Field {
    field(
        id,
        label,
        FieldKind::Choice {
            default: pairs.first().map(|(v, _)| (*v).to_owned()),
            options: options(pairs),
            display: ChoiceDisplay::Select,
        },
    )
}

/// Radio buttons; the first option is the default.
pub fn radio(id: &str, label: &str, pairs: &[(&str, &str)]) -> Field {
    field(
        id,
        label,
        FieldKind::Choice {
            default: pairs.first().map(|(v, _)| (*v).to_owned()),
            options: options(pairs),
            display: ChoiceDisplay::Radio,
        },
    )
}

/// One pick of the given kinds.
pub fn pick(id: &str, label: &str, accepts: &[SelectKind]) -> Field {
    field(
        id,
        label,
        FieldKind::Selection {
            accepts: accepts.to_vec(),
            multiple: false,
            max: Some(1),
        },
    )
}

/// One or more picks of the given kinds.
pub fn picks(id: &str, label: &str, accepts: &[SelectKind]) -> Field {
    field(
        id,
        label,
        FieldKind::Selection {
            accepts: accepts.to_vec(),
            multiple: true,
            max: None,
        },
    )
}

pub fn flag(id: &str, label: &str, default: bool) -> Field {
    field(id, label, FieldKind::Boolean { default }).optional()
}

pub fn text(id: &str, label: &str, placeholder: Option<&str>) -> Field {
    field(
        id,
        label,
        FieldKind::Text {
            default: None,
            placeholder: placeholder.map(str::to_owned),
        },
    )
}

pub fn list(id: &str, label: &str, item: Field) -> Field {
    field(
        id,
        label,
        FieldKind::List {
            item: Box::new(item.kind),
            min: Some(1),
            max: None,
        },
    )
}

/* ----------------------------------------------------------------------------------------- */
/* Predicates                                                                                */
/* ----------------------------------------------------------------------------------------- */

fn kinds_phrase(kinds: &[SelectKind]) -> String {
    let nouns: Vec<&str> = kinds.iter().map(|k| k.noun()).collect();
    match nouns.as_slice() {
        [] => "an item".to_owned(),
        [one] => format!("{} {one}", article(one)),
        [rest @ .., last] => {
            let first = rest[0];
            let middle = rest.join(", ");
            format!("{} {middle} or {last}", article(first))
        }
    }
}

fn article(noun: &str) -> &'static str {
    if noun.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    }
}

/// At least one selected item, each of one of `kinds`.
pub fn selected(kinds: &[SelectKind]) -> Predicate {
    Predicate::Selection {
        kinds: kinds.to_vec(),
        min: 1,
        max: None,
        reason: format!("Select {}", kinds_phrase(kinds)),
    }
}

/// Exactly one selected item of one of `kinds`.
pub fn selected_one(kinds: &[SelectKind]) -> Predicate {
    Predicate::Selection {
        kinds: kinds.to_vec(),
        min: 1,
        max: Some(1),
        reason: format!(
            "Select one {}",
            kinds_phrase(kinds)
                .split_once(' ')
                .map_or("item", |(_, rest)| rest)
        ),
    }
}

/// At least `min` selected items of `kinds`.
pub fn selected_at_least(kinds: &[SelectKind], min: u32) -> Predicate {
    let nouns: Vec<&str> = kinds.iter().map(|k| k.noun()).collect();
    Predicate::Selection {
        kinds: kinds.to_vec(),
        min,
        max: None,
        reason: format!("Select {min} or more items ({})", nouns.join(", ")),
    }
}

pub fn has(facet: Facet) -> Predicate {
    Predicate::Has {
        facet,
        reason: facet.reason().to_owned(),
    }
}

pub fn in_mode(modes: &[EditMode], reason: &str) -> Predicate {
    Predicate::Mode {
        modes: modes.to_vec(),
        reason: reason.to_owned(),
    }
}

pub fn all(predicates: Vec<Predicate>) -> Predicate {
    Predicate::All { all: predicates }
}

pub fn any(predicates: Vec<Predicate>) -> Predicate {
    Predicate::Any { any: predicates }
}

/* ----------------------------------------------------------------------------------------- */
/* Rows                                                                                      */
/* ----------------------------------------------------------------------------------------- */

pub fn c(id: &str) -> RowItem {
    RowItem::Command {
        command: id.to_owned(),
    }
}

/// A `▾` menu: the first entry runs on click.
pub fn menu(id: &str, label: &str, icon: &str, entries: Vec<MenuEntry>) -> RowItem {
    RowItem::Menu {
        id: id.to_owned(),
        label: label.to_owned(),
        icon: icon.to_owned(),
        entries,
    }
}

/// A menu entry that runs a command as is.
pub fn e(command: &str) -> MenuEntry {
    MenuEntry {
        command: command.to_owned(),
        label: None,
        values: BTreeMap::new(),
    }
}

/// A menu entry that starts `command` with one preset value.
pub fn preset(command: &str, label: &str, field: &str, value: &str) -> MenuEntry {
    let mut values = BTreeMap::new();
    values.insert(
        field.to_owned(),
        serde_json::Value::String(value.to_owned()),
    );
    MenuEntry {
        command: command.to_owned(),
        label: Some(label.to_owned()),
        values,
    }
}

/// A menu of presets of one command over the options of one of its choice fields.
pub fn variants(
    id: &str,
    label: &str,
    icon: &str,
    command: &str,
    field: &str,
    pairs: &[(&str, &str)],
) -> RowItem {
    menu(
        id,
        label,
        icon,
        pairs
            .iter()
            .map(|(value, label)| preset(command, label, field, value))
            .collect(),
    )
}

pub fn group(id: &str, label: &str, items: Vec<RowItem>) -> RowGroup {
    RowGroup {
        id: id.to_owned(),
        label: label.to_owned(),
        items,
    }
}

pub fn tab(id: &str, label: &str, groups: Vec<RowGroup>) -> DomainTab {
    DomainTab {
        id: id.to_owned(),
        label: label.to_owned(),
        groups,
    }
}

/// The options of a choice field, for building a menu that presets them.
pub fn choice_pairs(command: &CommandDescriptor, field: &str) -> Vec<(String, String)> {
    command
        .inputs
        .iter()
        .find(|f| f.id == field)
        .and_then(|f| match &f.kind {
            FieldKind::Choice { options, .. } => Some(
                options
                    .iter()
                    .map(|o| (o.value.clone(), o.label.clone()))
                    .collect(),
            ),
            _ => None,
        })
        .unwrap_or_default()
}
