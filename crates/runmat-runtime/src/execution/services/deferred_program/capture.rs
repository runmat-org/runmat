//! Shared selected code/capture payload, independent of invocation root roles.

use super::*;

pub struct DeferredSelectedCodeCapture {
    pub graph: PayloadGraph,
    pub required_schema: runmat_execution::value::GraphRequiredSchema,
    pub primary_unit: Arc<[u8]>,
    pub retained_units: Vec<Arc<[u8]>>,
    pub selected_foreign_policies: Vec<Option<runmat_execution::ForeignPolicyRef>>,
    pub class_dependencies: Vec<GraphClassDependency>,
    pub class_method_dependencies: Vec<GraphClassMethodDependency>,
    pub static_dependencies: Vec<runmat_execution::value::GraphStaticDependency>,
    pub selected_bundle_archives: Vec<Option<Arc<[u8]>>>,
    pub receipt: Rc<dyn GraphResultReceipt>,
}

impl DeferredSelectedCodeCapture {
    pub fn into_invocation(self, roots: GraphInvocationRoots) -> DeferredGraphCapture {
        DeferredGraphCapture {
            graph: self.graph,
            required_schema: self.required_schema,
            roots,
            primary_unit: self.primary_unit,
            retained_units: self.retained_units,
            selected_foreign_policies: self.selected_foreign_policies,
            class_dependencies: self.class_dependencies,
            class_method_dependencies: self.class_method_dependencies,
            static_dependencies: self.static_dependencies,
            selected_bundle_archives: self.selected_bundle_archives,
            receipt: self.receipt,
        }
    }

    /// A durable archive contains data, not the live receipt/source issuances.
    /// The restore producer re-admits all code and issues source owners normally.
    pub fn into_portable(self) -> runmat_execution::value::PortableSelectedCode {
        runmat_execution::value::PortableSelectedCode {
            schema_version: self.required_schema.version(),
            primary_unit: self.primary_unit.as_ref().to_vec(),
            retained_units: self
                .retained_units
                .iter()
                .map(|bytes| bytes.as_ref().to_vec())
                .collect(),
            graph: self.graph,
            selected_foreign_policies: self.selected_foreign_policies,
            class_dependencies: self.class_dependencies,
            class_method_dependencies: self.class_method_dependencies,
            static_dependencies: self.static_dependencies,
            selected_bundle_archives: self
                .selected_bundle_archives
                .into_iter()
                .map(|bytes| bytes.map(|bytes| bytes.as_ref().to_vec()))
                .collect(),
        }
    }
}

/// Fresh current-session captures plus their immutable source/code producer.
/// Neither member stores a previous invocation context or old service grant.
pub struct RestoredCallableGraph {
    pub values: crate::execution::RootedValues,
    pub program: DeferredGraphProgram,
}
