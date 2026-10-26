//! Documentation support shared by the meshing builtins.

use crate::*;

pub(in crate::catalog::entries::meshing) const MESHING_GUIDE: BuiltinDocumentationLink =
    BuiltinDocumentationLink {
        label: "Meshing",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/meshing"),
    };
pub(in crate::catalog::entries::meshing) const MESH_REFINEMENT_GUIDE: BuiltinDocumentationLink =
    BuiltinDocumentationLink {
        label: "Mesh refinement guide",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/guides/mesh-refinement",
        ),
    };
pub(in crate::catalog::entries::meshing) const LINKS: &[BuiltinDocumentationLink] =
    &[MESHING_GUIDE, MESH_REFINEMENT_GUIDE];

pub(in crate::catalog::entries::meshing) const BUILTINS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Meshing builtins",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/meshing/mod.rs",
    ),
};
pub(in crate::catalog::entries::meshing) const INSPECTION_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Mesh inspection projections",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/meshing/inspection.rs",
    ),
};
pub(in crate::catalog::entries::meshing) const OBJECT_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "meshing.Mesh and meshing.Evidence objects",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/meshing/object.rs",
    ),
};
pub(in crate::catalog::entries::meshing) const OPTIONS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Meshing option decoding",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/meshing/options.rs",
    ),
};

/// Fixture: the exact 1 m x 2 m x 3 m OCCT box solid shared with the FEA examples.
pub(in crate::catalog::entries::meshing) const BOX_FIXTURE: BuiltinExampleFixture =
    BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "meshing-box-brep",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[BuiltinFilesystemEntry::File {
            relative_path: "box.brep",
            content: BuiltinFixtureContent::Utf8(crate::fea_contract::example_fixtures::BOX_BREP),
        }],
    });

/// Meshing executes through the installed native meshing provider and imports
/// exact CAD through OCCT.
pub(in crate::catalog::entries::meshing) const NATIVE_MESHING: BuiltinExampleRequirements =
    BuiltinExampleRequirements {
        host: BuiltinExampleHostRequirement::NativeOnly,
        runtime: &[
            BuiltinRuntimeCapability::NativeCad,
            BuiltinRuntimeCapability::NativeMeshing,
        ],
        ..BuiltinExampleRequirements::NONE
    };
pub(in crate::catalog::entries::meshing) const NATIVE_MESHING_PLOT_CORE:
    BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::NativeOnly,
    runtime: &[
        BuiltinRuntimeCapability::PlotCore,
        BuiltinRuntimeCapability::NativeCad,
        BuiltinRuntimeCapability::NativeMeshing,
    ],
    ..BuiltinExampleRequirements::NONE
};

/// Program prefix: load the box and generate a coarse linear mesh. Every
/// inspection example starts from this so their outputs agree.
macro_rules! box_mesh_setup {
    () => {
        concat!(
            "part = geometry.load('box.brep', 'Units', 'm');\n",
            "solverMesh = meshing.generate(part, 'TargetSize', 0.5, 'MaximumDeviation', 0.05, 'ElementOrder', 'tet4', 'Seed', 42);\n",
        )
    };
}
pub(in crate::catalog::entries::meshing) use box_mesh_setup;

pub(in crate::catalog::entries::meshing) const fn native_example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    assertions: &'static str,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_FIXTURE,
        requirements: NATIVE_MESHING,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}

pub(in crate::catalog::entries::meshing) struct MeshingDocumentation {
    pub title: &'static str,
    pub summary: &'static str,
    pub description: &'static str,
    pub keywords: &'static [&'static str],
    pub related: &'static [&'static str],
    pub sections: &'static [BuiltinDocumentationSection],
    pub examples: &'static [BuiltinExample],
    pub faqs: &'static [BuiltinDocumentationFaq],
    pub implementation: &'static [BuiltinDocumentationLink],
    pub verification: &'static [BuiltinEvidenceReference],
}

pub(in crate::catalog::entries::meshing) const fn documentation(
    definition: MeshingDocumentation,
) -> BuiltinDocumentation {
    BuiltinDocumentation {
        authority: BuiltinDocumentationAuthority::Catalog,
        title: Some(definition.title),
        slug: Some(definition.title),
        summary: definition.summary,
        description: definition.description,
        keywords: definition.keywords,
        related: definition.related,
        sections: definition.sections,
        examples: definition.examples,
        example_exemption: None,
        faqs: definition.faqs,
        links: LINKS,
        media: &[],
        evidence: BuiltinDocumentationEvidence {
            implementation: definition.implementation,
            verification: definition.verification,
            notes: &[],
        },
        introduced: None,
        status: Some(BuiltinDocumentationStatus::Stable),
    }
}

pub(in crate::catalog::entries::meshing) const MESHING_ERRORS_PARAGRAPH: &str = "Every failure reports `RunMat:meshing:InvalidInput`, with a message that names the operation and the specific cause.";
