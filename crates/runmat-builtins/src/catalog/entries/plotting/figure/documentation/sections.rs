use crate::*;
pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`figure()` creates a new figure and returns RunMat's current host `f64` graphics-handle encoding.",
            "`figure(n)` selects figure `n`. If that handle does not exist yet, RunMat creates it.",
            "In RunMat extension mode, `figure('next')` creates a new figure instead of selecting an existing one.",
            "In RunMat extension mode, a positive scalar from any of the eight integer classes is read exactly as a figure identifier and must fit in `u32`.",
            "In RunMat extension mode, a positive single-precision scalar can select a figure. Logical and GPU-resident numeric targets reject without provider access.",
            "Typed-integer values for `Position`, `Color`, `NumberTitle`, `Visible`, `CurrentAxes`, and `SgTitle` are RunMat-only extensions and are rejected in compatibility mode before graphics state changes.",
            "Integer `Position` values cross an explicit host `f64` geometry boundary only after every authoritative integer element is proved exactly representable. Integer RGB values are read authoritatively and the normal `[0,1]` constraint admits only exact zero and one components.",
            "Integer `NumberTitle` and `Visible` values use an exact zero/nonzero predicate with no floating conversion. Integer `CurrentAxes` and `SgTitle` aliases must be exactly representable before lookup in RunMat's current `f64` graphics registry.",
            "`figure('Property', value, ...)` creates a fresh figure and applies the requested figure properties.",
            "`figure(n, 'Property', value, ...)` selects or creates figure `n` and applies the requested figure properties.",
            "`figure('Position', [left bottom width height])` stores MATLAB-style figure window geometry and makes it available through `get(f, 'Position')` and figure scene metadata.",
            "The selected figure becomes the target for subsequent plotting commands such as `plot`, `subplot`, `gca`, `clf`, and `hold`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Options",
        paragraphs: &[
            "`'Name'` sets the figure window name and can be inspected later with `get(f, 'Name')`.",
            "`'NumberTitle'` accepts logical values or MATLAB-style `'on'`/`'off'` strings.",
            "`'Visible'` accepts logical values or MATLAB-style `'on'`/`'off'` strings. Hidden figures keep their graphics state but do not open or update interactive windows until made visible again.",
            "`'Position'` accepts a finite four-element numeric vector `[left bottom width height]`; width and height must be positive.",
            "`'Color'` accepts supported color letters, full color names such as `'black'`, or RGB triples, and updates the figure background color.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and residency",
        paragraphs: &[
            "`figure` manages plotting state and handles. It does not execute numeric GPU kernels.",
            "`figure` only updates plotting state and returns RunMat's host-side `f64` handle encoding. Resident numeric values are not admitted as figure targets and are not gathered.",
            "`figure` performs no provider dispatch. GPU-resident arrays passed to later plotting calls may still remain on device when the renderer supports that path.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution, authority, and current limits",
        paragraphs: &[
            "The canonical binding uses the same runtime implementation in interpreter and compiled execution. Requested output arity is checked before this operation can mutate graphics state or gather coordinate data; discarding an output does not suppress the operation.",
            "Graphics handles encode local registry identities, not transferable figure, axes, or chart state. Copying a numeric handle to a worker, foreign runtime, or serialized value does not transfer its graphics authority. This host operation is a fusion boundary; no numeric GPU kernel or distributed implementation is supplied.",
            "Finite Managed allocation remains an L25 obligation: complete scratch, metadata, retained graphics-state, and output ownership are not yet quoted by these implementations. The shared graphics admission owner refuses Managed policy before unowned work, captures cancellation authority, and checks it before publication; that safety fence is not finite-allocation completion. Unbounded host scratch may still contain synchronous loops.",
        ],
    },
];
