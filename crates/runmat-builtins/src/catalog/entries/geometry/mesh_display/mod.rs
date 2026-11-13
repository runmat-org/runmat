//! `mesh(asset, ...)`: the method-only display binding for `geometry.Asset`.

use super::example_fixtures::{BOX_BREP, TRIANGLE_STL};
use crate::*;
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, EffectKind, ExecutionStackRequirement,
    NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact,
};

#[cfg(test)]
mod tests;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "asset",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "geometry.Asset returned by geometry.load.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Display options: Edges (logical, default true) and XRay (logical, default false).",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "figure",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle of the figure that shows the asset.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "figure = mesh(asset, Name, Value, ...)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const GEOMETRY_MESH_METHOD_ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.MESH.INVALID_INPUT",
    identifier: Some("RunMat:geometry:mesh:InvalidInput"),
    when: "The first argument is not a geometry.Asset, or the options are not Edges/XRay Name, Value pairs with logical values.",
    message: "mesh: invalid input",
};
pub const GEOMETRY_MESH_METHOD_ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.MESH.OPERATION_FAILED",
    identifier: Some("RunMat:geometry:mesh:OperationFailed"),
    when: "The asset has no renderable surface data or the host has no plotting support.",
    message: "mesh: operation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    GEOMETRY_MESH_METHOD_ERROR_INPUT,
    GEOMETRY_MESH_METHOD_ERROR_OPERATION,
];
pub const GEOMETRY_MESH_METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &ERRORS,
};
pub const GEOMETRY_MESH_METHOD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "mesh(asset, ...) accepts a geometry.Asset and logical display controls; it has no integer input role.",
    };

const PLOT_CORE: BuiltinExampleRequirements = BuiltinExampleRequirements {
    runtime: &[BuiltinRuntimeCapability::PlotCore],
    ..BuiltinExampleRequirements::NONE
};
const NATIVE_CAD_PLOT_CORE: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::NativeOnly,
    runtime: &[
        BuiltinRuntimeCapability::PlotCore,
        BuiltinRuntimeCapability::NativeCad,
    ],
    ..BuiltinExampleRequirements::NONE
};

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: None,
    slug: None,
    summary: "Display a loaded geometry asset with `mesh(asset)`.",
    description: "`mesh(asset)` and `mesh(asset, Name, Value, ...)` draw the render tessellation of a `geometry.Asset` in a CAD-style figure and return the figure handle. RunMat's object dispatch routes `mesh` on a `geometry.Asset` to this method-only binding; numeric `mesh(X, Y, Z)` is unchanged.",
    keywords: &["geometry", "mesh", "display", "cad", "preview", "plot"],
    related: &["geometry.load", "geometry.meshes", "geometry.region", "meshing.generate"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "The figure shows every surface mesh in the asset with equal axis scaling and a fixed three-quarter view. It is a view of the imported geometry only: it does not generate solver elements and does not change the asset. Use `meshing.generate` to discretize the geometry for analysis.",
                "Options are Name, Value pairs with case-insensitive names. `Edges` (logical, default `true`) draws the feature-edge overlay. `XRay` (logical, default `false`) renders faces translucent so internal faces and hidden edges are visible. Option values must be logical; numeric `1` or `0` is rejected.",
                "Display requires a host built with plotting support. On a host without it, the call fails with `RunMat:geometry:mesh:OperationFailed` instead of returning a handle.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Display the asset after `geometry.load` to confirm its orientation, scale, and topology before choosing regions with `geometry.region` and generating a solver mesh. An asset with no renderable surface mesh, such as a STEP file imported metadata-only without OCCT, cannot be displayed.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[
                "`RunMat:geometry:mesh:InvalidInput` reports a first argument that is not a `geometry.Asset`, options that are not Name, Value pairs, an unknown option name, or a non-logical option value.",
                "`RunMat:geometry:mesh:OperationFailed` reports an asset without renderable surface data or a host without plotting support.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "show-stl",
            title: "Display an imported STL",
            program: "part = geometry.load('part.stl');\nh = mesh(part);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: TRIANGLE_STL,
            requirements: PLOT_CORE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(isnumeric(h) && isscalar(h));",
            },
        },
        BuiltinExample {
            id: "xray-cad",
            title: "Inspect an exact CAD solid in x-ray mode",
            program: "part = geometry.load('box.brep', 'Units', 'm');\nh = mesh(part, 'Edges', true, 'XRay', true);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BOX_BREP,
            requirements: NATIVE_CAD_PLOT_CORE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(isnumeric(h) && isscalar(h));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `mesh(asset)` create a finite element mesh?",
            answer: "No. It displays the import tessellation. `meshing.generate(asset, ...)` creates the solver mesh, and `mesh(solverMesh)` displays that.",
        },
        BuiltinDocumentationFaq {
            question: "Can I pass `'Edges', 0`?",
            answer: "No. Option values must be logical, such as `true` or `false`.",
        },
    ],
    links: &[BuiltinDocumentationLink {
        label: "Geometry",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/geometry"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Geometry mesh display method",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/display.rs",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Asset method registration keeps exact public spellings",
                location: "crates/runmat-runtime/src/builtins/geometry/tests.rs::asset_method_registration_keeps_exact_public_spellings",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Geometry preview preserves assembly ownership and x-ray styling",
                location: "crates/runmat-runtime/src/geometry/tests.rs::geometry_preview_preserves_assembly_ownership_and_xray_styling",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    match request.arguments.first() {
        None => diagnostics.push(crate::catalog::argument_error(
            "RM-CATALOG-GEOMETRY-MESH-ARITY",
            "mesh requires a geometry.Asset",
            0,
        )),
        Some(asset) if !super::region::possible_asset(asset) => {
            diagnostics.push(crate::catalog::argument_error(
                "RM-CATALOG-GEOMETRY-MESH-ASSET",
                "mesh(asset, ...) requires a host geometry.Asset",
                0,
            ))
        }
        Some(_) => {}
    }
    let options = request.arguments.len().saturating_sub(1);
    if !options.is_multiple_of(2) {
        diagnostics.push(crate::catalog::argument_error(
            "RM-CATALOG-GEOMETRY-MESH-OPTIONS",
            "geometry mesh display options must be Name, Value pairs",
            request.arguments.len(),
        ));
    }
    for index in (1..request.arguments.len()).step_by(2) {
        if let Some(name) = request.literals.literal_string_at(index) {
            if name != "edges" && name != "xray" {
                diagnostics.push(crate::catalog::argument_error(
                    "RM-CATALOG-GEOMETRY-MESH-OPTION",
                    "geometry mesh display options are Edges and XRay",
                    index,
                ));
            }
        }
        if let Some(value) = request.arguments.get(index + 1) {
            if !matches!(value.kind, ValueKindFact::Logical | ValueKindFact::Unknown) {
                diagnostics.push(crate::catalog::argument_error(
                    "RM-CATALOG-GEOMETRY-MESH-LOGICAL",
                    "geometry mesh display option values must be logical",
                    index + 1,
                ));
            }
        }
    }
    let handle = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    crate::catalog::finish_fixed(entry, request, handle, diagnostics)
}

pub const GEOMETRY_MESH_METHOD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "__runmat_geometry_mesh",
    },
    category: "geometry/display",
    documentation: DOCUMENTATION,
    descriptor: &GEOMETRY_MESH_METHOD_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Plotting,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::UserInterface, EffectKind::MayThrow],
        capabilities: &[CapabilityRequirement::UserInterface],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::Host,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&GEOMETRY_MESH_METHOD_INTEGER_AUDIT),
    suppress_auto_output: true,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GEOMETRY_MESH_METHOD_CATALOG_ENTRY];
