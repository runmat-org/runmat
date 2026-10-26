//! Example fixture shared by geometry-bound model-object documentation.
//!
//! `block.obj` is a 0.2 m x 0.1 m x 0.1 m block whose six faces are named OBJ
//! groups, so `geometry.region(block, 'x_min')` and its siblings resolve to
//! distinct regions (`region_x_min`, ..., `region_z_max`). The mesh importer is
//! portable; no CAD kernel is required.

use crate::*;

pub const BLOCK_GEOMETRY: BuiltinExampleFixture =
    BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "fea-model-block",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[BuiltinFilesystemEntry::File {
            relative_path: "block.obj",
            content: BuiltinFixtureContent::Utf8(concat!(
                "# 0.2 m x 0.1 m x 0.1 m block; one group per face\n",
                "v 0 0 0\n",
                "v 0.2 0 0\n",
                "v 0.2 0.1 0\n",
                "v 0 0.1 0\n",
                "v 0 0 0.1\n",
                "v 0.2 0 0.1\n",
                "v 0.2 0.1 0.1\n",
                "v 0 0.1 0.1\n",
                "g x_min\n",
                "f 1 5 8\n",
                "f 1 8 4\n",
                "g x_max\n",
                "f 2 3 7\n",
                "f 2 7 6\n",
                "g y_min\n",
                "f 1 2 6\n",
                "f 1 6 5\n",
                "g y_max\n",
                "f 4 8 7\n",
                "f 4 7 3\n",
                "g z_min\n",
                "f 1 4 3\n",
                "f 1 3 2\n",
                "g z_max\n",
                "f 5 6 7\n",
                "f 5 7 8\n",
            )),
        }],
    });
