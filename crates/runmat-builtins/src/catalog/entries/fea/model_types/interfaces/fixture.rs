//! Two real named OBJ regions; no synthetic edits to immutable geometry values.
use crate::*;
pub(super) const PATCHES: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(
    BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId { local_name: "interface-patches" },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[BuiltinFilesystemEntry::File {
            relative_path: "interface-patches.obj",
            content: BuiltinFixtureContent::Utf8(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 0 1 1\ng left\nf 1 2 3\ng right\nf 4 5 6\n"
            ),
        }],
    });
