use crate::*;
pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`clf()` clears the current figure.",
            "`clf(handle)` clears the specified figure handle.",
            "`clf('all')` is a separately gated RunMat convenience that clears every known figure and returns the number of figures that were cleared; compatibility mode rejects it before plotting-state access.",
            "`clf(handle1, handle2, ...)` is a separately gated RunMat convenience that clears multiple figures and returns the number cleared. Compatibility syntax uses one figure object or figure array.",
            "`clf('reset')` is currently treated the same as plain `clf`.",
            "In RunMat mode, scalar and array figure numbers can use any of the eight integer classes. Values are read from authoritative integer storage and must be positive and no larger than `uint32`'s maximum.",
            "Typed integer figure numbers are rejected in compatibility mode because the public compatibility contract documents figure numbers but does not advertise integer classes. GPU-resident numeric targets are rejected without provider dispatch.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and residency",
        paragraphs: &[
            "`clf` clears plotting state only. It does not execute numeric GPU kernels.",
            "`clf` only mutates host plotting state. Figure-number controls must be host values; resident numeric targets are rejected without gathering.",
            "`clf` performs no provider dispatch; it clears figures from the plotting registry and renderer state.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution, authority, and current limits",
        paragraphs: &[
            "The canonical binding uses the same runtime implementation in interpreter and compiled execution. Requested output arity is checked before this operation can mutate graphics state or gather coordinate data; discarding an output does not suppress the operation.",
            "Graphics handles encode local registry identities, not transferable figure, axes, or chart state. Copying a numeric handle to a worker, foreign runtime, or serialized value does not transfer its graphics authority. This host operation is a fusion boundary; no numeric GPU kernel or distributed implementation is supplied.",
            "Finite Managed allocation remains an L25 obligation: complete scratch, metadata, retained graphics-state, and output ownership are not yet quoted by these implementations. The shared graphics admission owner refuses Managed policy before unowned work, captures cancellation authority, and checks it before publication; that safety fence is not finite-allocation completion. Unbounded host scratch may still contain synchronous loops.",
            "Plain clf and clf reset currently share the same full FigureState replacement. Figure properties are reset along with contents; the distinct MATLAB plain-versus-reset lifecycle is still an explicit compatibility obligation. The figure identifier remains alive. Clearing several figures is not a transactional rollback operation if an external authority changes between individual clears.",
        ],
    },
];
