use crate::*;
pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`gobjects()` returns a 1-by-1 placeholder handle array.",
            "`gobjects(n)` returns an n-by-n placeholder handle array.",
            "`gobjects(m,n,...)` returns a placeholder handle array with the requested dimensions.",
            "`gobjects([m n ...])` accepts a numeric size vector.",
            "Placeholder entries are not valid handles: `isgraphics` and `ishandle` return false for them.",
            "Assigned plotting handles retain normal numeric-handle fast paths and can be validated elementwise through `isgraphics` or `ishandle`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and residency",
        paragraphs: &[
            "`gobjects` allocates host-side graphics handle placeholders and does not execute numeric GPU kernels.",
            "`gobjects` always returns host-side placeholder handles because graphics state is host-managed.",
            "`gobjects` performs no provider dispatch and does not gather or upload GPU arrays.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution, authority, and current limits",
        paragraphs: &[
            "The canonical binding uses the same runtime implementation in interpreter and compiled execution. Requested output arity is checked before this operation can mutate graphics state or gather coordinate data; discarding an output does not suppress the operation.",
            "Graphics handles encode local registry identities, not transferable figure, axes, or chart state. Copying a numeric handle to a worker, foreign runtime, or serialized value does not transfer its graphics authority. This host operation is a fusion boundary; no numeric GPU kernel or distributed implementation is supplied.",
            "Finite Managed allocation remains an L25 obligation: complete scratch, metadata, retained graphics-state, and output ownership are not yet quoted by these implementations. The shared graphics admission owner refuses Managed policy before unowned work, captures cancellation authority, and checks it before publication; that safety fence is not finite-allocation completion. Unbounded host scratch may still contain synchronous loops.",
            "The result is a host double Tensor filled with NaN sentinels, including its scalar and empty forms. It is not a MATLAB GraphicsPlaceholder object. This representation frontier remains explicit; native integer size controls are structural inputs, not the result class. Negative dimensions clamp to zero; logical and GPU-resident controls reject without provider access.",
        ],
    },
];
