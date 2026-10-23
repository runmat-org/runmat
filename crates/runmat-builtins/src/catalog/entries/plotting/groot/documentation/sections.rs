use crate::*;
pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`groot()` returns scalar numeric handle `0`.",
            "`get(groot, 'Type')` returns `'root'`.",
            "`get(groot, 'CurrentFigure')` returns the current figure handle, or an empty handle array when no figure exists.",
            "`get(groot, 'Children')` returns currently open figure handles in stable ascending order.",
            "`set(groot, 'CurrentFigure', fig)` selects an existing figure as current.",
            "`set(groot, 'default...', value)` stores arbitrary root default properties for compatibility with scripts that configure graphics defaults.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and residency",
        paragraphs: &[
            "`groot()` returns the constant host scalar double zero without consulting or mutating a graphics registry. Separate get/set calls on that identity own their graphics-state effects; no provider dispatch, upload, or gather occurs.",
            "`groot` returns a host-side numeric graphics handle and does not interact with GPU residency.",
            "`groot()` returns the constant host scalar double zero without consulting or mutating a graphics registry. Separate get/set calls on that identity own their graphics-state effects; no provider dispatch, upload, or gather occurs.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution, authority, and current limits",
        paragraphs: &[
            "The canonical binding uses the same runtime implementation in interpreter and compiled execution. Requested output arity is checked before this operation can mutate graphics state or gather coordinate data; discarding an output does not suppress the operation.",
            "Graphics handles encode local registry identities, not transferable figure, axes, or chart state. Copying a numeric handle to a worker, foreign runtime, or serialized value does not transfer its graphics authority. This host operation is a fusion boundary; no numeric GPU kernel or distributed implementation is supplied.",
        ],
    },
];
