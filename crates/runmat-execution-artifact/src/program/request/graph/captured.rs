//! Portable selected code and shared lexical captures, independent of a job.

use super::*;

/// Canonical selected-unit bytes and one shared callable capture graph.
/// Stored coordinates are claims until artifact and VM admission both finish.
/// No session source ids, invoker, runtime, service or resource grant is stored.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedGraphCode(runmat_execution::value::PortableSelectedCode);

impl std::ops::Deref for CapturedGraphCode {
    type Target = runmat_execution::value::PortableSelectedCode;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Artifact-level admission, before current-session source issuance and exact
/// VM body/activation/static checks. Decoded foreign bundles remain paired with
/// this admission and their original selected-unit positions.
#[derive(Debug)]
pub struct ValidatedCapturedGraphCode {
    code: CapturedGraphCode,
    bundles: Vec<Option<ExecutionBundle>>,
    admission: Arc<()>,
}

/// Code whose ordered archives have been admitted and transferred to the host.
/// This is not an executable callable or a current runtime projection.
#[derive(Debug)]
pub struct ArchiveAdmittedCapturedGraphCode {
    code: CapturedGraphCode,
    admission: Arc<()>,
}

impl CapturedGraphCode {
    pub fn new(code: runmat_execution::value::PortableSelectedCode) -> Self {
        Self(code)
    }

    pub fn selected_source_map(&self) -> ArtifactResult<ExecutableSourceMapV2> {
        self.selected_code().selected_source_map()
    }

    fn selected_code(&self) -> selection::SelectedCode<'_> {
        selection::SelectedCode {
            primary_unit: &self.primary_unit,
            retained_units: &self.retained_units,
            graph: &self.graph,
            class_dependencies: &self.class_dependencies,
            class_method_dependencies: &self.class_method_dependencies,
            static_dependencies: &self.static_dependencies,
            selected_bundle_archives: &self.selected_bundle_archives,
            selected_foreign_policies: &self.selected_foreign_policies,
            primary_policy: None,
        }
    }

    pub fn into_validated(self) -> ArtifactResult<ValidatedCapturedGraphCode> {
        if GraphRequiredSchema::from_version(self.schema_version)
            .is_none_or(|schema| schema < self.graph.required_schema())
        {
            return Err(ArtifactError::Invalid(
                "captured graph schema cannot admit its selected value features".into(),
            ));
        }
        if self.graph.roots.is_empty()
            || self.graph.roots.len() > runmat_execution::value::PortableSelectedCode::MAX_CALLABLE_ROOTS
            || self.graph.roots.iter().any(|root| {
                !matches!(
                    self.graph.nodes.get(root.0 as usize),
                    Some(GraphNode::Callable(callable)) if matches!(callable.owner, GraphCallableOwner::Bound(_))
                )
            })
        {
            return Err(ArtifactError::Invalid(
                "captured graph roots must select actual callable nodes".into(),
            ));
        }
        let selected = self.selected_code();
        let _sources = selected.selected_source_map()?;
        let _host = selected.selected_unit_host_requirements()?;
        let bundles = selected.selected_bundle_admission()?;
        self.graph
            .validate_selected_unit_references(self.retained_units.len() + 1)
            .map_err(super::super::contract_error)?;
        selected.validate_class_dependencies()?;
        selected.validate_static_dependencies()?;
        if self.graph.nodes.len() as u64 > ValueLimits::default().max_nodes {
            return Err(ArtifactError::Limit(
                "captured graph exceeds its node bound".into(),
            ));
        }
        Ok(ValidatedCapturedGraphCode {
            code: self,
            bundles,
            admission: Arc::new(()),
        })
    }
}

impl ValidatedCapturedGraphCode {
    pub fn code(&self) -> &CapturedGraphCode {
        &self.code
    }

    pub fn selected_bundles(&self) -> &[Option<ExecutionBundle>] {
        &self.bundles
    }

    pub fn into_admitted_with_bundles(
        self,
    ) -> (
        ArchiveAdmittedCapturedGraphCode,
        Vec<AdmittedGraphUnitBundle>,
    ) {
        let bundles = admitted_bundles(self.bundles, &self.admission);
        (
            ArchiveAdmittedCapturedGraphCode {
                code: self.code,
                admission: self.admission,
            },
            bundles,
        )
    }
}

impl ArchiveAdmittedCapturedGraphCode {
    pub fn code(&self) -> &CapturedGraphCode {
        &self.code
    }

    pub fn consume_bundle_for(
        &self,
        unit_index: usize,
        bundle: AdmittedGraphUnitBundle,
    ) -> ArtifactResult<Option<ExecutionBundle>> {
        consume_admitted_bundle(
            self.code.retained_units.len() + 1,
            &self.admission,
            unit_index,
            bundle,
        )
    }
}
