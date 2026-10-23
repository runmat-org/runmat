use crate::ir::NativeAssembly;
use crate::{NativeCodegenError, NativeCodegenResult};

fn analysis_bytes(analysis: &runmat_mir::analysis::AnalysisStore) -> NativeCodegenResult<Vec<u8>> {
    analysis
        .canonical_bytes()
        .map_err(|error| NativeCodegenError::new("native.analysis.encoding", error))
}

pub(super) fn analysis_digest(
    analysis: &runmat_mir::analysis::AnalysisStore,
) -> NativeCodegenResult<runmat_execution::Digest> {
    Ok(runmat_execution::Digest::sha256(&analysis_bytes(analysis)?))
}

pub(super) fn verify_analysis_component(
    analysis: &runmat_mir::analysis::AnalysisStore,
    manifest: &runmat_execution::ExecutableUnitManifest,
) -> NativeCodegenResult<runmat_execution::Digest> {
    let digest = analysis_digest(analysis)?;
    let admitted = manifest
        .components
        .iter()
        .find(|component| component.kind == runmat_execution::ExecutableComponentKind::Analysis)
        .ok_or_else(|| {
            NativeCodegenError::new(
                "native.lowering.analysis_component",
                "executable manifest has no admitted AnalysisStore component",
            )
        })?;
    if admitted.digest != digest {
        return Err(NativeCodegenError::new(
            "native.lowering.analysis_component",
            "supplied AnalysisStore bytes differ from the admitted executable component",
        ));
    }
    Ok(digest)
}

pub(super) fn project_analysis(
    analysis: &runmat_mir::analysis::AnalysisStore,
    mir: &runmat_mir::MirAssembly,
) -> NativeCodegenResult<runmat_mir::analysis::AnalysisStore> {
    let owners = mir
        .bodies
        .keys()
        .map(|function| {
            analysis
                .function_for_local(*function)
                .map(|facts| (*function, facts.function))
                .ok_or_else(|| {
                    NativeCodegenError::new(
                        "native.lowering.analysis_projection",
                        "retained MIR body has no admitted analysis owner",
                    )
                    .at_function(*function)
                })
        })
        .collect::<NativeCodegenResult<std::collections::BTreeMap<_, _>>>()?;
    analysis
        .project_product_functions(&owners)
        .map_err(|error| NativeCodegenError::new("native.lowering.analysis_projection", error))
}

/// Binds operation-local Native contracts to the independently admitted
/// AnalysisStore. A digest carried by Native IR alone never grants authority.
pub fn verify_against_analysis(
    assembly: &NativeAssembly,
    manifest: &runmat_execution::ExecutableUnitManifest,
    analysis: &runmat_mir::analysis::AnalysisStore,
) -> NativeCodegenResult<()> {
    super::verify_against_manifest(assembly, manifest)?;
    analysis.revision.validate_current().map_err(|error| {
        NativeCodegenError::new("native.ir.analysis_revision", error.to_string())
    })?;
    if analysis_digest(analysis)? != assembly.analysis_component {
        return Err(NativeCodegenError::new(
            "native.ir.analysis_component",
            "Native IR does not belong to the supplied admitted AnalysisStore",
        ));
    }
    let owners = assembly
        .functions
        .iter()
        .map(|function| (function.id, function.portable_id))
        .collect::<std::collections::BTreeMap<_, _>>();
    if owners.len() != assembly.functions.len() {
        return Err(NativeCodegenError::new(
            "native.ir.analysis_projection",
            "Native functions do not have unique analysis owners",
        ));
    }
    let projected = analysis
        .project_product_functions(&owners)
        .map_err(|error| NativeCodegenError::new("native.ir.analysis_projection", error))?;
    if analysis_digest(&projected)? != assembly.analysis_projection {
        return Err(NativeCodegenError::new(
            "native.ir.analysis_projection",
            "Native IR analysis projection differs from the admitted AnalysisStore",
        ));
    }
    let projected_contracts = projected
        .operation_contracts
        .iter()
        .map(|contract| (&contract.site, contract))
        .collect::<std::collections::BTreeMap<_, _>>();
    let native_contracts = assembly
        .functions
        .iter()
        .flat_map(|function| &function.operation_contracts)
        .map(|contract| (&contract.site, contract))
        .collect::<std::collections::BTreeMap<_, _>>();
    let native_contract_count = assembly
        .functions
        .iter()
        .map(|function| function.operation_contracts.len())
        .sum::<usize>();
    if projected_contracts.len() != projected.operation_contracts.len()
        || native_contracts.len() != native_contract_count
        || native_contracts.len() != projected.operation_contracts.len()
        || native_contracts.iter().any(|(site, native)| {
            projected_contracts
                .get(site)
                .is_none_or(|analysis| *analysis != *native)
        })
    {
        return Err(NativeCodegenError::new(
            "native.ir.analysis_contracts",
            "Native operation contracts differ from the admitted AnalysisStore projection",
        ));
    }
    Ok(())
}
