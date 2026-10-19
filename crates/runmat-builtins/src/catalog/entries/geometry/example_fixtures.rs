//! Filesystem fixtures shared by geometry documentation examples.

use crate::*;

/// One-triangle ASCII STL; imports through the portable mesh importer.
pub(super) const TRIANGLE_STL: BuiltinExampleFixture =
    BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "triangle-stl",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[BuiltinFilesystemEntry::File {
            relative_path: "part.stl",
            content: BuiltinFixtureContent::Utf8(
                "solid tri\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid tri\n",
            ),
        }],
    });

/// The repository's exact 1 m x 2 m x 3 m OCCT solid. Importing it requires a
/// native host with the OCCT exact-CAD importer.
pub(super) const BOX_BREP: BuiltinExampleFixture =
    BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "box-brep",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[BuiltinFilesystemEntry::File {
            relative_path: "box.brep",
            content: BuiltinFixtureContent::Utf8(crate::fea_contract::example_fixtures::BOX_BREP),
        }],
    });

/// Native host boundary for examples that import exact CAD through OCCT.
pub(super) const NATIVE_CAD: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::NativeOnly,
    runtime: &[BuiltinRuntimeCapability::NativeCad],
    ..BuiltinExampleRequirements::NONE
};
