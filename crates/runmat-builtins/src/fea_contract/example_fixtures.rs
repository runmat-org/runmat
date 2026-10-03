//! Public scripting inputs for the closed-volume FEA example chain.
//! The corresponding analysis operation test uses the same cube topology and
//! linear-static profile; executable catalog qualification is still required.

use crate::*;

pub const CLOSED_CUBE_STUDY: BuiltinExampleFixture =
    BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "closed-cube-study",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[
            BuiltinFilesystemEntry::File {
                relative_path: "cube.fea",
                content: BuiltinFixtureContent::Utf8(
                    "version: 1\nkind: study\nid: closed_cube_static\ngeometry:\n  path: cube.stl\n  units: meter\nmodel:\n  profile: linear_static_structural\nrun:\n  backend: cpu\n",
                ),
            },
            BuiltinFilesystemEntry::File {
                relative_path: "cube.stl",
                content: BuiltinFixtureContent::Utf8(concat!(
                    "solid cube\n",
                    "facet normal 0 0 -1\nouter loop\nvertex 0 0 0\nvertex 1 1 0\nvertex 1 0 0\nendloop\nendfacet\n",
                    "facet normal 0 0 -1\nouter loop\nvertex 0 0 0\nvertex 0 1 0\nvertex 1 1 0\nendloop\nendfacet\n",
                    "facet normal 0 0 1\nouter loop\nvertex 0 0 1\nvertex 1 0 1\nvertex 1 1 1\nendloop\nendfacet\n",
                    "facet normal 0 0 1\nouter loop\nvertex 0 0 1\nvertex 1 1 1\nvertex 0 1 1\nendloop\nendfacet\n",
                    "facet normal 0 -1 0\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 1 0 1\nendloop\nendfacet\n",
                    "facet normal 0 -1 0\nouter loop\nvertex 0 0 0\nvertex 1 0 1\nvertex 0 0 1\nendloop\nendfacet\n",
                    "facet normal 1 0 0\nouter loop\nvertex 1 0 0\nvertex 1 1 0\nvertex 1 1 1\nendloop\nendfacet\n",
                    "facet normal 1 0 0\nouter loop\nvertex 1 0 0\nvertex 1 1 1\nvertex 1 0 1\nendloop\nendfacet\n",
                    "facet normal 0 1 0\nouter loop\nvertex 1 1 0\nvertex 0 1 0\nvertex 0 1 1\nendloop\nendfacet\n",
                    "facet normal 0 1 0\nouter loop\nvertex 1 1 0\nvertex 0 1 1\nvertex 1 1 1\nendloop\nendfacet\n",
                    "facet normal -1 0 0\nouter loop\nvertex 0 1 0\nvertex 0 0 0\nvertex 0 0 1\nendloop\nendfacet\n",
                    "facet normal -1 0 0\nouter loop\nvertex 0 1 0\nvertex 0 0 1\nvertex 0 1 1\nendloop\nendfacet\n",
                    "endsolid cube\n"
                )),
            },
        ],
    });
