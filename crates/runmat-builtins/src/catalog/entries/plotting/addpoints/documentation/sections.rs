use crate::*;
pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`addpoints(an,x,y)` appends 2-D points to an animated line.",
            "`addpoints(an,x,y,z)` appends 3-D points. An empty 2-D animated line is promoted to a 3-D line when Z data is first added, unless the handle has marker metadata that the 3-D renderer cannot preserve.",
            "Each X, Y, or Z coordinate independently accepts single, double, or any of the eight integer classes. Integer storage is read authoritatively before coordinates enter the host graphics representation.",
            "Coordinate inputs must be scalars or vectors with matching element counts.",
            "Logical coordinate values are not numeric integer coordinates and are rejected.",
            "If the target is already 3-D, callers must provide X, Y, and Z coordinates.",
            "The function mutates graphics state and has no MATLAB output.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Options",
        paragraphs: &[
            "The first argument must be an `animatedline` handle.",
            "X, Y, and optional Z coordinates may be single, double, or any integer class, including GPU-resident scalars or vectors.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and residency",
        paragraphs: &[
            "`addpoints` mutates graphics handle state. GPU-resident coordinate inputs are accepted through the plotting input path and gathered before appending.",
            "`addpoints` gathers GPU-resident coordinate vectors before appending to host-side animated-line graphics state.",
            "The operation preserves graphics handle identity while replacing the stored line data.",
            "The updated line continues through RunMat's existing line or line3 rendering path.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution, authority, and current limits",
        paragraphs: &[
            "The canonical binding uses the same runtime implementation in interpreter and compiled execution. Requested output arity is checked before this operation can mutate graphics state or gather coordinate data; discarding an output does not suppress the operation.",
            "Graphics handles encode local registry identities, not transferable figure, axes, or chart state. Copying a numeric handle to a worker, foreign runtime, or serialized value does not transfer its graphics authority. This host operation is a fusion boundary; no numeric GPU kernel or distributed implementation is supplied.",
            "Finite Managed allocation remains an L25 obligation: complete scratch, metadata, retained graphics-state, and output ownership are not yet quoted by these implementations. The shared graphics admission owner refuses Managed policy before unowned work, captures cancellation authority, and checks it before publication; that safety fence is not finite-allocation completion. Unbounded host scratch may still contain synchronous loops.",
            "The original rooted argument owner survives every coordinate gather await. Actual invocation context and full figure/axes ownership are retained. Cancellation or authority failure after gather rejects before appending. Trailing singleton dimensions are tolerated, but non-singleton pages are not a scalar/vector form.",
        ],
    },
];
