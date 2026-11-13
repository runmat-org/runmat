//! Hidden binding behind `mesh(run)` for FEA run, results, and field objects.

use crate::fea_contract::{
    descriptors::FEA_MESH_DESCRIPTOR,
    doc_links,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    result_inference::{infer_result, FeaResultKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_result(request, entry, FeaResultKind::Mesh)
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("__runmat_fea_mesh"),
    slug: Some("__runmat_fea_mesh"),
    summary: "Internal binding that implements mesh(run) for FEA run, results, and field objects.",
    description: "`figure = mesh(runOrResultsOrField, Name, Value, ...)` dispatches here for `fea.RunResult`, `fea.StepRunResult`, `fea.Results`, and `fea.Field` objects. It draws the exact solver mesh used by that run and returns the figure handle. Call `mesh` rather than this name; numeric `mesh(X, Y, Z)` is unaffected.",
    keywords: &["fea", "mesh", "solver mesh", "visualization", "internal"],
    related: &["fea.run", "fea.results", "fea.field", "__runmat_fea_plot", "mesh"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The object must carry the study it came from and a run id; objects returned by `fea.run`, `fea.stepResult`, `fea.results`, and `fea.field` do. A multi-step `fea.RunResult` needs a step selected first.",
        "Display options at the end of the argument list, names case-insensitive with punctuation ignored: `Source` (`\"solver\"` by default, or `\"cad\"`, `\"cad_reference\"`, or `\"auto\"`) chooses what to draw; `Edges` (logical, default `true`) draws element edges. Options change only the figure.",
        "The overload requires the runtime's plotting support; without it the call fails with `RunMat:fea:OperationFailed`.",
    ] }],
    examples: &[],
    example_exemption: Some("Hidden method binding; it is reached through mesh(run), shown in the meshing and FEA guides."),
    faqs: &[],
    links: &[doc_links::MESHING],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[doc_links::FEA_VISUALIZATION_SOURCE],
        verification: &[
            doc_links::PLOT_MESH_TEST,
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Mesh requests default to solver edges and accept CAD reference", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::mesh_request_defaults_to_solver_edges_and_accepts_cad_reference" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

pub const FEA_MESH_METHOD_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "__runmat_fea_mesh",
        documentation: DOCUMENTATION,
        descriptor: &FEA_MESH_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::Figure,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_MESH_METHOD_CATALOG_ENTRY];
