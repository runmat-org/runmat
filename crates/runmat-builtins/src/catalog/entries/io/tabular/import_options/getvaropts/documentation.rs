use super::super::shared::{SOURCE, TABLE_TEST};
use crate::*;

pub(super) const GETVAROPTS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("getvaropts"), slug: Some("getvaropts"),
    summary: "Inspect selected variable import controls.",
    description: "`getvaropts` returns a typed variable-options object for one selected variable or an object array for multiple variables.",
    keywords: &["getvaropts", "setvaropts", "VariableOptions"],
    related: &["setvaropts", "setvartype", "detectImportOptions"],
    sections: &[BuiltinDocumentationSection { heading: "Selected variable metadata", paragraphs: &[
        "A returned object exposes Name, Type, and supported per-variable controls. A multiple selection returns an object array. The full MATLAB family of type-specific VariableOptions subclasses is not yet represented.",
    ] }],
    examples: &[BuiltinExample {
        id: "inspect", title: "Inspect one variable",
        program: "opts = spreadsheetImportOptions('NumVariables',2);\nv = getvaropts(opts,'Var2');",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(v.Name,'Var2'));" },
    }],
    example_exemption: None, faqs: &[], links: &[SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[TABLE_TEST], notes: &["Source packet only; runtime execution proof remains pending."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
