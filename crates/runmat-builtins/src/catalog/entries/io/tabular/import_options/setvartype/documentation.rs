use super::super::shared::{SOURCE, TABLE_TEST};
use crate::*;

pub(super) const SETVARTYPE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("setvartype"), slug: Some("setvartype"),
    summary: "Change the import type of selected variables.",
    description: "`setvartype` changes selected variable types in a typed import-options object and returns a new object. The existing table conversion owner performs the requested typed conversion.",
    keywords: &["setvartype", "VariableTypes", "detectImportOptions", "readtable"],
    related: &["setvaropts", "getvaropts", "readtable"],
    sections: &[BuiltinDocumentationSection { heading: "Type selection", paragraphs: &[
        "Select variables by exact name, one-based index, or logical mask, then supply one supported type or one type per variable. Native integer classes use the existing exact text parser; unsupported types fail before file I/O.",
    ] }],
    examples: &[BuiltinExample {
        id: "numeric-type", title: "Set a variable type",
        program: "opts = spreadsheetImportOptions('NumVariables',2);\nopts = setvartype(opts,'Var2','double');\nv = getvaropts(opts,'Var2');",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(v.Type,'double'));" },
    }],
    example_exemption: None, faqs: &[], links: &[SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[TABLE_TEST], notes: &["Type-specific public VariableOptions subclasses remain a compatibility gap."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
