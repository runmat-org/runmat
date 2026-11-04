//! The CAD dialects' statement forms: features in the call form, with each kind's readable
//! arguments, sketches as blocks and assembly occurrences as `use` statements.
//!
//! ```text
//! feature body = extrude(profile = sketch.s1, distance = 10 mm)
//! feature cut = extrude(profile = sketch.slot, through_all, mode = cut)
//! feature h = hole(face = feature.body.face.end, at = { a = point(60 mm, 10 mm, 10 mm) },
//!                  diameter = 4 mm, through_all)
//! feature round = fillet(edges = [edges_between(feature.body.face.side(s1.l1),
//!                                               feature.body.face.end)], radius = 1.5 mm)
//! feature mid = plane.offset(base = origin.xy, distance = 20 mm)
//! ```
//!
//! Every kind has the call form `<kind>(<field> = <value>, …)`; a kind with a second tag is
//! written `<kind>.<variant>(…)`. The kinds below also take shorter arguments that stand for a
//! nested value (`distance = 10 mm` for `extent = blind(distance = 10 mm)`); each is a
//! one-to-one rewrite, so the long form reads too. The record form `extrude { … }` still reads.

use super::{occurrence, sketch};
use runmat_design_syntax::ast;
use runmat_design_syntax::document::{BlockEntry, EntryForms};
use runmat_design_syntax::json::{Json, LowerError, ValueSyntax};

/// Fields that hold internally tagged values whose tag key is the field's name, anywhere in
/// the CAD models: `mode = cut`, `extent = blind(distance = 10 mm)`.
const TAG_FIELDS: &[&str] = &[
    "extent", "mode", "layout", "seeds", "profile", "end", "relief", "rule", "motion", "hand",
];

/// Fields of one entry kind's arguments that hold tagged values under another key.
const SCOPED_TAG_FIELDS: &[(&str, &str, &str)] = &[
    ("extrude", "from", "from"),
    ("extrude", "second", "extent"),
    ("hole", "form", "hole"),
    ("chamfer", "form", "chamfer"),
    ("draft", "kind", "draft"),
];

pub fn tag_field(scope: Option<&str>, key: &str) -> Option<&'static str> {
    if let Some(scope) = scope {
        if let Some((_, _, tag)) = SCOPED_TAG_FIELDS
            .iter()
            .find(|(kind, field, _)| *kind == scope && *field == key)
        {
            return Some(tag);
        }
    }
    TAG_FIELDS.iter().find(|field| **field == key).copied()
}

/// Feature kinds with a second tag, written `<kind>.<variant>(…)`.
const SUBTAGS: &[(&str, &str)] = &[
    ("plane", "plane"),
    ("axis", "axis"),
    ("point", "point_def"),
    ("frame", "frame"),
    ("boolean", "kind"),
    ("body_op", "op"),
    ("surface", "kind"),
    ("sheet_metal", "sheet_metal"),
    ("direct", "direct"),
    ("weldment", "weldment"),
    ("mold", "mold"),
    ("curve", "curve"),
];

pub struct CadForms;

type Fields = Vec<(String, Json)>;

fn get<'a>(fields: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn object(entries: Vec<(&str, Json)>) -> Json {
    Json::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

/// `{ <tag key> = <tag>, rest… }` as `(tag, rest)`.
fn tagged<'a>(value: &'a Json, key: &str) -> Option<(&'a str, &'a [(String, Json)])> {
    let Json::Object(entries) = value else {
        return None;
    };
    let ((first, Json::String(tag)), rest) = entries.split_first()? else {
        return None;
    };
    (first == key).then_some((tag.as_str(), rest))
}

/// Keys of `rest`, in order.
fn keys(rest: &[(String, Json)]) -> Vec<&str> {
    rest.iter().map(|(k, _)| k.as_str()).collect()
}

/// One kind's short arguments: each stands for (part of) one field's tagged value.
struct Shorthand {
    /// The field the arguments fill.
    field: &'static str,
    /// Its tag key.
    tag: &'static str,
    /// `(variant, the variant's fields, the arguments that write it)`. An argument written
    /// `"x"` takes the field `x`'s value; `"!x"` is a flag (`x = true`) with no field.
    forms: &'static [(
        &'static str,
        &'static [&'static str],
        &'static [&'static str],
    )],
}

const EXTRUDE: Shorthand = Shorthand {
    field: "extent",
    tag: "extent",
    forms: &[
        ("blind", &["distance"], &["distance"]),
        ("symmetric", &["distance"], &["distance", "!symmetric"]),
        ("through_all", &[], &["!through_all"]),
        ("up_to_next", &[], &["!up_to_next"]),
        ("up_to", &["face"], &["up_to"]),
    ],
};

const HOLE_END: Shorthand = Shorthand {
    field: "end",
    tag: "end",
    forms: &[
        ("blind", &["depth"], &["depth"]),
        ("through_all", &[], &["!through_all"]),
        ("up_to_next", &[], &["!up_to_next"]),
    ],
};

const CHAMFER: Shorthand = Shorthand {
    field: "form",
    tag: "chamfer",
    forms: &[
        ("equal", &["distance"], &["distance"]),
        (
            "distances",
            &["distance", "distance2"],
            &["distance", "distance2"],
        ),
        (
            "distance_angle",
            &["distance", "angle"],
            &["distance", "angle"],
        ),
        ("vertex", &["distances"], &["distances"]),
    ],
};

const SEEDS: Shorthand = Shorthand {
    field: "seeds",
    tag: "seeds",
    forms: &[
        ("features", &["features"], &["features"]),
        ("bodies", &["bodies"], &["bodies"]),
    ],
};

fn shorthands(tag: &str) -> &'static [Shorthand] {
    match tag {
        "extrude" => &[EXTRUDE],
        "hole" => &[HOLE_END],
        "chamfer" => &[CHAMFER],
        "pattern" | "mirror" => &[SEEDS],
        _ => &[],
    }
}

impl Shorthand {
    /// Every argument word this shorthand reads.
    fn words(&self) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = Vec::new();
        for (_, _, arguments) in self.forms {
            for argument in *arguments {
                let word = argument.trim_start_matches('!');
                if !out.contains(&word) {
                    out.push(word);
                }
            }
        }
        out
    }

    /// The field's value as short arguments, when one form writes it.
    fn sugar(&self, value: &Json) -> Option<Fields> {
        let (variant, rest) = tagged(value, self.tag)?;
        let (_, fields, arguments) = self
            .forms
            .iter()
            .find(|(v, fields, _)| *v == variant && keys(rest) == *fields)?;
        let mut out = Vec::new();
        let mut taken = fields.iter();
        for argument in *arguments {
            match argument.strip_prefix('!') {
                Some(flag) => out.push((flag.to_string(), Json::Bool(true))),
                None => {
                    let field = taken.next()?;
                    out.push((argument.to_string(), get(rest, field)?.clone()));
                }
            }
        }
        Some(out)
    }

    /// Reads the short arguments back into the field. Returns the field, where it goes (the
    /// position of its first argument) and the arguments' model paths.
    #[allow(clippy::type_complexity)]
    fn desugar(
        &self,
        arguments: &mut Fields,
    ) -> Result<Option<(Json, usize, Vec<(String, String)>)>, String> {
        let words = self.words();
        let given: Vec<(usize, String)> = arguments
            .iter()
            .enumerate()
            .filter(|(_, (k, _))| words.contains(&k.as_str()))
            .map(|(i, (k, _))| (i, k.clone()))
            .collect();
        if given.is_empty() {
            return Ok(None);
        }
        if get(arguments, self.field).is_some() {
            return Err(format!(
                "`{}` and `{}` both set `{}`; give one",
                self.field, given[0].1, self.field
            ));
        }
        let names: Vec<&str> = given.iter().map(|(_, k)| k.as_str()).collect();
        let form = self.forms.iter().find(|(_, _, written)| {
            let mut a: Vec<&str> = written.iter().map(|w| w.trim_start_matches('!')).collect();
            let mut b = names.clone();
            a.sort_unstable();
            b.sort_unstable();
            a == b
        });
        let Some((variant, fields, written)) = form else {
            return Err(format!("`{}` do not go together here", names.join("`, `")));
        };
        let mut value = vec![(self.tag.to_string(), Json::String(variant.to_string()))];
        let mut moved = Vec::new();
        let mut field_names = fields.iter();
        for argument in *written {
            let (word, flag) = match argument.strip_prefix('!') {
                Some(flag) => (flag, true),
                None => (*argument, false),
            };
            let item = get(arguments, word).cloned().unwrap_or(Json::Null);
            if flag {
                if item != Json::Bool(true) {
                    return Err(format!("`{word}` is a flag; write it alone"));
                }
                moved.push((word.to_string(), self.field.to_string()));
            } else {
                let field = field_names.next().expect("one field per value argument");
                moved.push((word.to_string(), format!("{}.{field}", self.field)));
                value.push((field.to_string(), item));
            }
        }
        let at = given[0].0;
        let mut index = 0;
        arguments.retain(|_| {
            let keep = !given.iter().any(|(i, _)| *i == index);
            index += 1;
            keep
        });
        Ok(Some((Json::Object(value), at, moved)))
    }
}

/// A hole's positions as `at = { <key> = <point>, … }`.
fn positions_sugar(value: &Json) -> Option<Json> {
    let Json::Array(items) = value else {
        return None;
    };
    let mut out: Vec<(String, Json)> = Vec::new();
    for item in items {
        let key = item.get("key")?.as_str()?;
        let at = item.get("at")?;
        if item.as_object_len() != 2 || out.iter().any(|(k, _)| k == key) {
            return None;
        }
        out.push((key.to_string(), at.clone()));
    }
    Some(Json::Object(out))
}

trait ObjectLen {
    fn as_object_len(&self) -> usize;
}

impl ObjectLen for Json {
    fn as_object_len(&self) -> usize {
        match self {
            Json::Object(entries) => entries.len(),
            _ => 0,
        }
    }
}

impl EntryForms for CadForms {
    fn subtag(&self, word: &str, tag: &str) -> Option<&'static str> {
        if word != "feature" {
            return None;
        }
        SUBTAGS.iter().find(|(t, _)| *t == tag).map(|(_, key)| *key)
    }

    fn desugar(
        &self,
        word: &str,
        tag: &str,
        mut arguments: Vec<(String, Json)>,
    ) -> Result<(Vec<(String, Json)>, Vec<(String, String)>), String> {
        if word != "feature" {
            return Ok((arguments, Vec::new()));
        }
        let mut moved = Vec::new();
        for shorthand in shorthands(tag) {
            if let Some((value, at, paths)) = shorthand.desugar(&mut arguments)? {
                arguments.insert(
                    at.min(arguments.len()),
                    (shorthand.field.to_string(), value),
                );
                moved.extend(paths);
            }
        }
        if tag == "hole" {
            if let Some(position) = arguments.iter().position(|(k, _)| k == "at") {
                if get(&arguments, "positions").is_some() {
                    return Err("`at` and `positions` both set the hole positions".into());
                }
                let Json::Object(points) = &arguments[position].1 else {
                    return Err("a hole's `at` is `{ <key> = <point>, … }`".into());
                };
                let mut list = Vec::new();
                for (index, (key, point)) in points.iter().enumerate() {
                    moved.push((format!("at.{key}"), format!("positions[{index}].at")));
                    list.push(object(vec![
                        ("key", Json::String(key.clone())),
                        ("at", point.clone()),
                    ]));
                }
                moved.push(("at".into(), "positions".into()));
                arguments[position] = ("positions".into(), Json::Array(list));
            }
        }
        Ok((arguments, moved))
    }

    fn sugar(&self, word: &str, tag: &str, fields: &[(String, Json)]) -> Option<Fields> {
        if word != "feature" {
            return None;
        }
        let shorthands = shorthands(tag);
        let taken: Vec<&str> = shorthands.iter().flat_map(|s| s.words()).collect();
        if fields
            .iter()
            .any(|(k, _)| taken.contains(&k.as_str()) || k == "at")
        {
            return None;
        }
        let mut out = Vec::new();
        for (key, value) in fields {
            if let Some(shorthand) = shorthands.iter().find(|s| s.field == key) {
                if let Some(arguments) = shorthand.sugar(value) {
                    out.extend(arguments);
                    continue;
                }
            }
            if tag == "hole" && key == "positions" {
                if let Some(points) = positions_sugar(value) {
                    out.push(("at".into(), points));
                    continue;
                }
            }
            out.push((key.clone(), value.clone()));
        }
        Some(out)
    }

    fn lower_block(
        &self,
        block: &ast::BlockStatement,
        hook: &dyn ValueSyntax,
    ) -> Option<Result<BlockEntry, LowerError>> {
        use runmat_design_syntax::ast::Headed;
        match block.keyword().as_str() {
            "sketch" => Some(sketch::lower(block, hook)),
            _ => None,
        }
    }

    fn lower_use(
        &self,
        declaration: &ast::UseDeclaration,
        hook: &dyn ValueSyntax,
    ) -> Option<Result<BlockEntry, LowerError>> {
        Some(occurrence::lower(declaration, hook))
    }

    fn print_statement(
        &self,
        word: &str,
        name: &str,
        fields: &[(String, Json)],
        hook: &dyn ValueSyntax,
    ) -> Option<String> {
        match word {
            "feature" => {
                let (tag, _) = tagged(get(fields, "kind")?, "feature")?;
                (tag == "sketch").then(|| sketch::print(name, fields, hook))?
            }
            "occurrence" => occurrence::print(name, fields, hook),
            _ => None,
        }
    }
}

/// The surface forms' own words, for the highlighter (`scripts/docs/fea-grammar.mjs` reads
/// these tables).
pub mod words {
    /// Statement words besides the collections' words.
    pub const STATEMENTS: &[&str] = &["sketch", "use"];
    /// Sketch entities, sketch constraints, reference and datum forms, and short arguments.
    pub const OPERATIONS: &[&str] = &[
        "point",
        "line",
        "arc",
        "circle",
        "construction",
        "coincident",
        "horizontal",
        "vertical",
        "parallel",
        "perpendicular",
        "tangent",
        "equal",
        "fix",
        "distance",
        "length",
        "radius",
        "angle",
        "limit",
        "edges_between",
        "vertex_of",
        "piece",
        "origin",
        "through_all",
        "up_to_next",
        "up_to",
        "symmetric",
    ];
}
