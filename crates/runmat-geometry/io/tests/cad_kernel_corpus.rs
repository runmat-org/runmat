//! K1's robustness harness: runmat-cad's naming corpus, regeneration, family, Features-tab and
//! DFM suites, run unchanged on the OCCT kernel. The same oracles and expected diagnostics
//! that hold on the grid test kernel must hold on exact geometry.
//!
//! Per-operation outcomes are appended to `$RUNMAT_K1_STATS` (one `op<TAB>outcome` line per
//! kernel call) when it is set, for failure-rate reports.

// Each included runmat-cad suite declares its own `mod common;`, as it does in runmat-cad.
#![allow(clippy::duplicate_mod)]

#[path = "cad_corpus/kernel_under_test.rs"]
mod kernel_under_test;

#[path = "cad_corpus/k1_ops.rs"]
mod k1_ops;

#[path = "cad_corpus/k1_sheet.rs"]
mod k1_sheet;

#[path = "cad_corpus/k1_sheet_flanges.rs"]
mod k1_sheet_flanges;

#[path = "cad_corpus/k1_sheet_corners.rs"]
mod k1_sheet_corners;

#[path = "../../../runmat-cad/tests/common/mod.rs"]
mod common;

#[path = "../../../runmat-cad/tests/naming_kinds.rs"]
mod naming_kinds;

#[path = "../../../runmat-cad/tests/naming_corpus.rs"]
mod naming_corpus;

#[path = "../../../runmat-cad/tests/regen.rs"]
mod regen;

#[path = "../../../runmat-cad/tests/families.rs"]
mod families;

#[path = "../../../runmat-cad/tests/features_tab.rs"]
mod features_tab;

#[path = "../../../runmat-cad/tests/dfm_analysis.rs"]
mod dfm_analysis;
