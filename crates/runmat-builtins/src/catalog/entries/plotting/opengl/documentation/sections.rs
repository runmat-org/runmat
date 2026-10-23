use crate::*;
pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Syntax",
        paragraphs: &[
            "status = opengl",
            "info = opengl('info')",
            "data = opengl('data')",
            "status = opengl('hardware')",
            "status = opengl('hardwarebasic')",
            "status = opengl('software')",
            "status = opengl('save', mode)",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Compatibility notes",
        paragraphs: &[
            "`opengl('info')` and `opengl('data')` return a struct with MATLAB-style fields such as `Version`, `Vendor`, `Renderer`, `HardwareSupportLevel`, `Software`, and `MaxTextureSize`.",
            "Mode-selection commands are deterministic compatibility operations; they do not replace the `runmat-plot` backend or install a native OpenGL context.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Arguments",
        paragraphs: &[
            "command",
            "text scalar",
            "Renderer command: `info`, `data`, `hardware`, `hardwarebasic`, `software`, `save`, `autoselect`, or `neverselect`. `opengl('save', mode)` accepts `hardware`, `hardwarebasic`, `software`, `autoselect`, or `none`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and residency",
        paragraphs: &[
            "`opengl` reports renderer metadata only. It does not affect gpuArray residency or provider dispatch.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution, authority, and current limits",
        paragraphs: &[
            "The canonical binding uses the same runtime implementation in interpreter and compiled execution. Requested output arity is checked before this operation can mutate graphics state or gather coordinate data; discarding an output does not suppress the operation.",
            "Graphics handles encode local registry identities, not transferable figure, axes, or chart state. Copying a numeric handle to a worker, foreign runtime, or serialized value does not transfer its graphics authority. This host operation is a fusion boundary; no numeric GPU kernel or distributed implementation is supplied.",
            "Finite Managed allocation remains an L25 obligation: complete scratch, metadata, retained graphics-state, and output ownership are not yet quoted by these implementations. The shared graphics admission owner refuses Managed policy before unowned work, captures cancellation authority, and checks it before publication; that safety fence is not finite-allocation completion. Unbounded host scratch may still contain synchronous loops.",
            "Information fields are fixed compatibility values, not measurements or guarantees about installed hardware, texture limits, smoothing, transparency, or an active native OpenGL context. Mode commands return deterministic status but do not persist a renderer setting, inspect a provider, or switch a backend.",
        ],
    },
];
