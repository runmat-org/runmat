//! Hidden binding behind `plot(field, ...)` for `fea.Field` objects.

use crate::fea_contract::{
    descriptors::FEA_PLOT_DESCRIPTOR,
    doc_links,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    result_inference::{infer_result, FeaResultKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_result(request, entry, FeaResultKind::Plot)
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("__runmat_fea_plot"),
    slug: Some("__runmat_fea_plot"),
    summary: "Internal binding that implements plot(field) for fea.Field objects.",
    description: "`figure = plot(field, Name, Value, ...)` dispatches here when `field` is a `fea.Field`. It renders the field on its solver mesh and returns the figure handle. Call `plot` rather than this name.",
    keywords: &["fea", "plot", "field", "visualization", "internal"],
    related: &["fea.field", "__runmat_fea_mesh", "plot"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The field must carry its run id and the study it came from, which `fea.field` attaches when given a run, a step run, or results derived from `fea.run`. The field's id and frequency sample select the stored values to draw.",
        "Display options, matched case-insensitively at the end of the argument list: `Edges` (logical, default `false`) draws solver-mesh edges; `Overlay` (`\"auto\"`, `\"solver\"`, or `\"cad\"`, default `\"auto\"`) chooses the reference geometry drawn with the field; `Deformed` (logical, default `true`) draws the field on the deformed shape when displacement is available. Options change only the figure.",
        "Fields whose values live on an accelerator are rejected rather than gathered implicitly. The overload requires the runtime's plotting support; without it the call fails with `RunMat:fea:OperationFailed`.",
    ] }],
    examples: &[],
    example_exemption: Some("Hidden method binding; it is reached through plot(field), whose usage is shown by fea.field."),
    faqs: &[],
    links: &[doc_links::OUTPUTS_AND_FIELDS],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[doc_links::FEA_VISUALIZATION_SOURCE],
        verification: &[doc_links::PLOT_MESH_TEST],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

pub const FEA_PLOT_METHOD_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "__runmat_fea_plot",
        documentation: DOCUMENTATION,
        descriptor: &FEA_PLOT_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::Figure,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_PLOT_METHOD_CATALOG_ENTRY];
