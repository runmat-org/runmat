//! Embedded reference material for Codex prompts.

pub const REFERENCE_FILE_PATHS: &[&str] = &[
    "docs/ARCHITECTURE.md",
    "crates/runmat-accelerate/README.md",
    "crates/runmat-accelerate/src/lib.rs",
    "crates/runmat-accelerate/src/native_auto.rs",
    "crates/runmat-accelerate/src/fusion.rs",
    "crates/runmat-builtins/src/lib.rs",
    "crates/runmat-runtime/BUILTIN_PACKAGING.md",
    "crates/runmat-runtime/src/builtins/common/spec.rs",
    "crates/runmat-runtime/src/builtins/math/reduction/sum.rs",
    "crates/runmat-runtime/src/builtins/math/reduction/mean.rs",
    "crates/runmat-runtime/src/builtins/math/trigonometry/sin.rs",
];

const ARCHITECTURE_MD: &str = include_str!("../../../../docs/ARCHITECTURE.md");
const ACCELERATE_README: &str = include_str!("../../../../crates/runmat-accelerate/README.md");
const ACCELERATE_LIB: &str = include_str!("../../../../crates/runmat-accelerate/src/lib.rs");
const ACCELERATE_NATIVE_AUTO: &str = include_str!("../../../../crates/runmat-accelerate/src/native_auto.rs");
const ACCELERATE_FUSION: &str = include_str!("../../../../crates/runmat-accelerate/src/fusion.rs");
const RUNMAT_BUILTINS_LIB: &str = include_str!("../../../../crates/runmat-builtins/src/lib.rs");
const BUILTIN_PACKAGING: &str = include_str!("../../../../crates/runmat-runtime/BUILTIN_PACKAGING.md");
const COMMON_SPEC: &str = include_str!("../../../../crates/runmat-runtime/src/builtins/common/spec.rs");
const SUM_REDUCTION: &str = include_str!("../../../../crates/runmat-runtime/src/builtins/math/reduction/sum.rs");
const MEAN_REDUCTION: &str = include_str!("../../../../crates/runmat-runtime/src/builtins/math/reduction/mean.rs");
const SIN_TRIGONOMETRY: &str = include_str!("../../../../crates/runmat-runtime/src/builtins/math/trigonometry/sin.rs");

pub const REFERENCE_SECTIONS: &[(&str, &str)] = &[
    (REFERENCE_FILE_PATHS[0], ARCHITECTURE_MD),
    (REFERENCE_FILE_PATHS[1], ACCELERATE_README),
    (REFERENCE_FILE_PATHS[2], ACCELERATE_LIB),
    (REFERENCE_FILE_PATHS[3], ACCELERATE_NATIVE_AUTO),
    (REFERENCE_FILE_PATHS[4], ACCELERATE_FUSION),
    (REFERENCE_FILE_PATHS[5], RUNMAT_BUILTINS_LIB),
    (REFERENCE_FILE_PATHS[6], BUILTIN_PACKAGING),
    (REFERENCE_FILE_PATHS[7], COMMON_SPEC),
    (REFERENCE_FILE_PATHS[8], SUM_REDUCTION),
    (REFERENCE_FILE_PATHS[9], MEAN_REDUCTION),
    (REFERENCE_FILE_PATHS[10], SIN_TRIGONOMETRY),
];

pub fn reference_sections() -> &'static [(&'static str, &'static str)] {
    REFERENCE_SECTIONS
}
