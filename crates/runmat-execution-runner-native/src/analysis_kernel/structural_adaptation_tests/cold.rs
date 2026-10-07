//! Reopen authenticated original object closures after the producing store is closed.
use super::*;
use runmat_analysis_execution::{
    import_analysis_objects, AnalysisHostWorkload, PreparedAnalysisObjects,
};
use runmat_execution::value::ValueRefKind;

pub(super) struct OriginalObjects {
    invocation: PreparedAnalysisObjects,
    mesh: PreparedAnalysisObjects,
    result: PreparedAnalysisObjects,
    receipt: PreparedAnalysisObjects,
}
impl OriginalObjects {
    pub(super) fn source<'a>(
        &'a self,
        host: &'a AnalysisHostWorkload,
    ) -> StructuralAdaptationNativeSource<'a> {
        StructuralAdaptationNativeSource {
            host,
            invocation: &self.invocation,
            mesh: &self.mesh,
            result: &self.result,
            receipt: &self.receipt,
        }
    }
}
pub(super) fn reload(
    store: &FilesystemObjectStore,
    source: StructuralAdaptationNativeSource<'_>,
) -> OriginalObjects {
    let load = |objects: &PreparedAnalysisObjects, kind| {
        let references = objects
            .references(&source.host.artifact_access, kind)
            .unwrap();
        import_analysis_objects(
            store,
            &references[0],
            objects.manifest.kind,
            objects.manifest.logical_identity,
            objects.manifest.payload_digest,
            &source.host.artifact_access,
            ObjectInventoryLimits::default(),
        )
        .unwrap()
    };
    OriginalObjects {
        invocation: load(source.invocation, ValueRefKind::DriverObject),
        mesh: load(source.mesh, ValueRefKind::DriverObject),
        result: load(source.result, ValueRefKind::ResultObject),
        receipt: load(source.receipt, ValueRefKind::ResultObject),
    }
}
