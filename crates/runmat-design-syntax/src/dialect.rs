//! The dialect registry: one grammar, one dialect per file kind.
//!
//! `.fea`, `.part`, `.assembly`, `.drawing`, `.circuit` and `.circuit.layout` share the lexer,
//! the expression grammar, the lossless tree, the formatter and the highlighter. A dialect says
//! which words head its file-level declarations, which words open clauses in block headers,
//! and which generic statement forms it takes. Domain vocabulary (`extrude`, `mate`, `net`)
//! belongs to each front end's registry, never to the grammar.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dialect {
    Fea,
    Part,
    Assembly,
    Drawing,
    Circuit,
    CircuitLayout,
}

/// A statement form that lives in the grammar, shared by every dialect that takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenericForm {
    /// `runmat <version>` and `std <version>`.
    VersionHeader,
    /// `use <kind> <name> from "<path>" [@ "<pin>"] [as <alias>] [{ … }]`, `use "<folder>"`.
    Use,
    /// `constrain <expr> [= <expr>]`.
    Constrain,
    /// `when <condition> { … }`.
    When,
    /// `publish <name> [= <value>]`.
    Publish,
    /// `check <name> [= <value> | { … }]`.
    Check,
    /// `property <name> [: <type>] = <value>`.
    Property,
    /// `link <kind> <from> to <target>`.
    Link,
    /// `solved { … }`.
    Solved,
}

impl GenericForm {
    pub const ALL: &'static [GenericForm] = &[
        GenericForm::VersionHeader,
        GenericForm::Use,
        GenericForm::Constrain,
        GenericForm::When,
        GenericForm::Publish,
        GenericForm::Check,
        GenericForm::Property,
        GenericForm::Link,
        GenericForm::Solved,
    ];

    /// The contextual word that opens the form. These are words, not reserved keywords: a
    /// `.fea` file may still name things `check` or `link`.
    pub const fn word(self) -> &'static str {
        match self {
            GenericForm::VersionHeader => "runmat",
            GenericForm::Use => "use",
            GenericForm::Constrain => "constrain",
            GenericForm::When => "when",
            GenericForm::Publish => "publish",
            GenericForm::Check => "check",
            GenericForm::Property => "property",
            GenericForm::Link => "link",
            GenericForm::Solved => "solved",
        }
    }
}

/// The second version-header word.
pub const STD_HEADER_WORD: &str = "std";

/// Words that start a value statement in every design dialect.
pub const VALUE_WORDS: &[&str] = &["let", "input", "property"];

/// Reserved spellings in design values.
pub const VALUE_LITERALS: &[&str] = &["true", "false", "none"];

/// Words that open a clause in a block header (`sketch profile on plane.base { … }`).
pub const CLAUSE_WORDS: &[&str] = &["on", "of", "from", "over", "at", "in", "to", "as", "for"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DialectSpec {
    pub dialect: Dialect,
    /// The file extension, without the dot (`circuit.layout` for board files).
    pub extension: &'static str,
    pub label: &'static str,
    /// Words that head a file-level declaration.
    pub declarations: &'static [&'static str],
    /// Words that head a file-level value declaration, `<word> <name> = <value>`
    /// (`footprint lcc18 = { … }` in a `.circuit` library).
    pub value_declarations: &'static [&'static str],
    /// The generic forms this dialect takes.
    pub forms: &'static [GenericForm],
    /// The `.fea` statement shapes and diagnostics, unchanged until the `.fea` semantic
    /// migration (M1+) moves them onto the generic forms.
    pub fea_statements: bool,
}

const DESIGN_FORMS: &[GenericForm] = GenericForm::ALL;

pub const DIALECTS: &[DialectSpec] = &[
    DialectSpec {
        dialect: Dialect::Fea,
        extension: "fea",
        label: "FEA study",
        declarations: &[
            "geometry",
            "mesh",
            "material",
            "model",
            "study",
            "sweep",
            "execution",
            "table",
        ],
        value_declarations: &[],
        forms: &[],
        fea_statements: true,
    },
    DialectSpec {
        dialect: Dialect::Part,
        extension: "part",
        label: "Part",
        declarations: &["part"],
        value_declarations: &[],
        forms: DESIGN_FORMS,
        fea_statements: false,
    },
    DialectSpec {
        dialect: Dialect::Assembly,
        extension: "assembly",
        label: "Assembly",
        declarations: &["assembly"],
        value_declarations: &[],
        forms: DESIGN_FORMS,
        fea_statements: false,
    },
    DialectSpec {
        dialect: Dialect::Drawing,
        extension: "drawing",
        label: "Drawing",
        declarations: &["drawing"],
        value_declarations: &[],
        forms: DESIGN_FORMS,
        fea_statements: false,
    },
    DialectSpec {
        dialect: Dialect::Circuit,
        extension: "circuit",
        label: "Circuit",
        declarations: &["circuit", "sheet"],
        value_declarations: &["footprint", "symbol", "part"],
        forms: DESIGN_FORMS,
        fea_statements: false,
    },
    DialectSpec {
        dialect: Dialect::CircuitLayout,
        extension: "circuit.layout",
        label: "Board layout",
        declarations: &["layout"],
        value_declarations: &[],
        forms: DESIGN_FORMS,
        fea_statements: false,
    },
];

impl Dialect {
    pub const ALL: &'static [Dialect] = &[
        Dialect::Fea,
        Dialect::Part,
        Dialect::Assembly,
        Dialect::Drawing,
        Dialect::Circuit,
        Dialect::CircuitLayout,
    ];

    pub fn spec(self) -> &'static DialectSpec {
        DIALECTS
            .iter()
            .find(|spec| spec.dialect == self)
            .expect("every dialect has a spec")
    }

    /// The dialect of a path, by its extension (`board.circuit.layout` is a layout).
    pub fn for_path(path: &str) -> Option<Dialect> {
        let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
        DIALECTS
            .iter()
            .filter(|spec| {
                name.len() > spec.extension.len() + 1
                    && name.ends_with(spec.extension)
                    && name[..name.len() - spec.extension.len()].ends_with('.')
            })
            .max_by_key(|spec| spec.extension.len())
            .map(|spec| spec.dialect)
    }

    pub fn extension(self) -> &'static str {
        self.spec().extension
    }

    pub fn takes(self, form: GenericForm) -> bool {
        self.spec().forms.contains(&form)
    }

    pub fn is_declaration_word(self, word: &str) -> bool {
        self.spec().declarations.contains(&word)
    }

    /// The file stem a design declaration must be named after (`parts/bracket.part` →
    /// `bracket`).
    pub fn stem(self, path: &str) -> String {
        let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
        name.strip_suffix(self.extension())
            .and_then(|rest| rest.strip_suffix('.'))
            .unwrap_or(name)
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialects_follow_the_file_extension() {
        assert_eq!(Dialect::for_path("a/b/beam.fea"), Some(Dialect::Fea));
        assert_eq!(Dialect::for_path("bracket.part"), Some(Dialect::Part));
        assert_eq!(Dialect::for_path("rig.assembly"), Some(Dialect::Assembly));
        assert_eq!(Dialect::for_path("rig.drawing"), Some(Dialect::Drawing));
        assert_eq!(Dialect::for_path("ctl.circuit"), Some(Dialect::Circuit));
        assert_eq!(
            Dialect::for_path("ctl.circuit.layout"),
            Some(Dialect::CircuitLayout)
        );
        assert_eq!(Dialect::for_path("notes.md"), None);
        assert_eq!(Dialect::for_path(".part"), None);
        assert_eq!(Dialect::Part.stem("parts/bracket.part"), "bracket");
        assert_eq!(Dialect::CircuitLayout.stem("ctl.circuit.layout"), "ctl");
    }

    #[test]
    fn only_the_design_dialects_take_the_generic_forms_until_the_fea_migration() {
        for dialect in Dialect::ALL {
            let spec = dialect.spec();
            assert_eq!(spec.fea_statements, *dialect == Dialect::Fea);
            assert_eq!(spec.forms.is_empty(), *dialect == Dialect::Fea);
        }
    }
}
