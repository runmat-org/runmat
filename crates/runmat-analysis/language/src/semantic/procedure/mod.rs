mod accuracy;
mod assessment;
mod header;
mod initial_conditions;
mod load_cases;
mod lower;
mod modal_reduction;
mod output;
mod prestressed_harmonic;
mod requirements;
mod study;
mod thermal_preload;
mod values;

use super::CheckedFeaDeclaration;
use crate::{BoundFeaProgram, FeaDiagnostic, FeaSymbolKey, FeaSymbolKind, LogicalSourcePath};
use runmat_analysis_core::{AuthoredModel, AuthoredStudy};
use runmat_geometry_core::GeometryDefinition;
use runmat_meshing_core::MeshDefinition;
use std::collections::BTreeMap;

pub(crate) use output::{field_location, field_reference_type};
pub(crate) use study::symbol_key as resolve_symbol_key;
pub(crate) use values::canonical_scalar;

pub fn lower_studies(
    bound: &BoundFeaProgram,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    models: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel>,
    geometries: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), GeometryDefinition>,
    meshes: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), MeshDefinition>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredStudy> {
    let context = study::StudyLowering {
        bound,
        models,
        geometries,
        meshes,
    };
    declarations
        .iter()
        .filter(|((_, key), _)| key.kind == FeaSymbolKind::Study)
        .filter_map(|(identity, declaration)| {
            study::lower(&context, declaration, diagnostics).map(|study| (identity.clone(), study))
        })
        .collect()
}
