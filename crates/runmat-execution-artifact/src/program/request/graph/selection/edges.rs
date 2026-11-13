//! Exact unit-qualified class and sealed-static edge checks.

use super::*;

impl SelectedCode<'_> {
    pub(crate) fn validate_class_dependencies(&self) -> ArtifactResult<()> {
        if self.class_dependencies.len() > runmat_execution::value::GRAPH_MAX_SELECTED_UNITS {
            return Err(ArtifactError::Limit(
                "graph class dependencies exceed their shared bound".into(),
            ));
        }
        let units = std::iter::once(self.primary_unit)
            .chain(self.retained_units.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let callers = self
            .graph
            .nodes
            .iter()
            .filter_map(|node| match node {
                GraphNode::Callable(callable) => match callable.owner {
                    GraphCallableOwner::Bound(owner) => Some(owner),
                    _ => None,
                },
                GraphNode::SelectedObject(object) => Some(object.constructor),
                _ => None,
            })
            .collect::<HashSet<_>>();
        let mut declarations = HashMap::new();
        let mut selected = HashSet::new();
        for edge in self.class_dependencies {
            let caller = usize::from(edge.caller.unit.0);
            let declaration = usize::from(edge.declaration_unit.0);
            if caller >= units.len() || declaration == 0 || declaration >= units.len() {
                return Err(ArtifactError::Invalid(
                    "graph class dependency selects an absent unit".into(),
                ));
            }
            if !callers.contains(&edge.caller) {
                return Err(ArtifactError::Invalid(
                    "graph class dependency has no selected callable".into(),
                ));
            }
            let (digest, class) = match declarations.entry(declaration) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let bytes = units[declaration];
                    let digest = runmat_execution::Digest::sha256(bytes);
                    let unit = ExecutableUnitEnvelope::from_canonical_bytes(bytes)
                        .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                    let class = match unit.manifest.identity.validated_entrypoint() {
                        Ok(runmat_execution::ValidatedExecutableEntrypoint::DeclarationOnly(
                            class,
                        )) => class,
                        _ => {
                            return Err(ArtifactError::Invalid(
                                "graph class dependency is not a declaration-only product".into(),
                            ));
                        }
                    };
                    entry.insert((digest, class))
                }
            };
            if *digest != edge.declaration_digest {
                return Err(ArtifactError::Invalid(
                    "graph class dependency differs from its admitted bytes".into(),
                ));
            }
            if !selected.insert((edge.caller, class.clone())) {
                return Err(ArtifactError::Invalid(
                    "graph callable repeats a class declaration dependency".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn validate_static_dependencies(&self) -> ArtifactResult<()> {
        let bytes = std::iter::once(self.primary_unit)
            .chain(self.retained_units.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let units = bytes
            .iter()
            .map(|bytes| {
                ExecutableUnitEnvelope::from_canonical_bytes(bytes)
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))
            })
            .collect::<ArtifactResult<Vec<_>>>()?;
        let revisions = units
            .iter()
            .map(|unit| {
                unit.manifest
                    .identity
                    .program
                    .identity_digest()
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))
            })
            .collect::<ArtifactResult<Vec<_>>>()?;
        let digests = bytes
            .iter()
            .map(runmat_execution::Digest::sha256)
            .collect::<Vec<_>>();
        let selected_code_bytes = bytes.iter().try_fold(0_usize, |total, bytes| {
            total
                .checked_add(bytes.len())
                .ok_or_else(|| ArtifactError::Limit("graph selected code bytes overflow".into()))
        })?;
        self.validate_class_method_dependencies(&revisions, &digests, selected_code_bytes)?;
        let sealed_bytes = units
            .iter()
            .flat_map(|unit| &unit.manifest.optional_sections)
            .filter(|section| section.name == STATIC_IMPORT_SECTION)
            .try_fold(0_usize, |total, section| {
                total.checked_add(section.payload.len()).ok_or_else(|| {
                    ArtifactError::Limit("graph sealed static section bytes overflow".into())
                })
            })?;
        if self.static_dependencies.len() > sealed_bytes {
            return Err(ArtifactError::Limit(
                "graph static edges exceed their sealed section bound".into(),
            ));
        }
        let mut content_digests = vec![None; units.len()];
        let mut edges = HashMap::<(GraphStaticCaller, GraphStaticReferenceSite), _>::new();
        for edge in self.static_dependencies {
            let caller_index = match edge.caller {
                GraphStaticCaller::Script(unit) => usize::from(unit.0),
                GraphStaticCaller::Function(owner) => usize::from(owner.unit.0),
            };
            let target_index = usize::from(edge.target.unit.0);
            if caller_index >= units.len() || target_index >= units.len() {
                return Err(ArtifactError::Invalid(
                    "graph static dependency selects an absent unit".into(),
                ));
            }
            if let GraphStaticCaller::Function(owner) = edge.caller {
                if owner.origin.program != revisions[caller_index] {
                    return Err(ArtifactError::Invalid(
                        "graph static caller differs from its admitted revision".into(),
                    ));
                }
            }
            if edge.target.origin.program != revisions[target_index]
                || edge.target_unit_digest != digests[target_index]
            {
                return Err(ArtifactError::Invalid(
                    "graph static target differs from its admitted unit".into(),
                ));
            }
            if edges.insert((edge.caller, edge.site), edge).is_some() {
                return Err(ArtifactError::Invalid(
                    "graph static dependency repeats a caller site".into(),
                ));
            }
        }
        for (index, unit) in units.iter().enumerate() {
            let unit_id = runmat_execution::value::GraphProgramUnitId(
                u16::try_from(index).map_err(|_| {
                    ArtifactError::Limit("graph static caller unit exceeds its ordinal ABI".into())
                })?,
            );
            let Some(section) = unit
                .manifest
                .optional_sections
                .iter()
                .find(|section| section.name == STATIC_IMPORT_SECTION)
            else {
                continue;
            };
            let imports = decode_static_imports(section)
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
            for import in imports {
                let caller = match import.caller {
                    ExecutableStaticCaller::Script => GraphStaticCaller::Script(unit_id),
                    ExecutableStaticCaller::Function(origin) => {
                        if origin.program != revisions[index] {
                            return Err(ArtifactError::Invalid(
                                "sealed static caller differs from its admitted revision".into(),
                            ));
                        }
                        GraphStaticCaller::Function(runmat_execution::value::GraphBoundOwner {
                            unit: unit_id,
                            origin,
                        })
                    }
                };
                let edge = edges.remove(&(caller, import.site)).ok_or_else(|| {
                    ArtifactError::Invalid("sealed static import has no selected graph edge".into())
                })?;
                let target_index = usize::from(edge.target.unit.0);
                let content = match content_digests[target_index] {
                    Some(content) => content,
                    None => {
                        let content = selected_unit_content_digest(&units[target_index])
                            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                        content_digests[target_index] = Some(content);
                        content
                    }
                };
                if edge.target.origin != import.target
                    || edge.target_body_digest != import.target_body
                    || content != import.target_content
                {
                    return Err(ArtifactError::Invalid(
                        "graph static edge differs from its sealed target".into(),
                    ));
                }
            }
        }
        if !edges.is_empty() {
            return Err(ArtifactError::Invalid(
                "graph static edge has no sealed caller site".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn validate_class_method_dependencies(
        &self,
        revisions: &[runmat_execution::Digest],
        unit_digests: &[runmat_execution::Digest],
        selected_code_bytes: usize,
    ) -> ArtifactResult<()> {
        if self.class_method_dependencies.len() > selected_code_bytes {
            return Err(ArtifactError::Limit(
                "graph class method edges exceed their selected code bound".into(),
            ));
        }
        let mut seen = HashSet::new();
        for edge in self.class_method_dependencies {
            let caller = usize::from(edge.caller.unit.0);
            let target = usize::from(edge.target.unit.0);
            if caller >= revisions.len() || target >= revisions.len() {
                return Err(ArtifactError::Invalid(
                    "graph class method edge selects an absent unit".into(),
                ));
            }
            if edge.caller.origin.program != revisions[caller]
                || edge.target.origin.program != revisions[target]
                || edge.target_unit_digest != unit_digests[target]
            {
                return Err(ArtifactError::Invalid(
                    "graph class method edge differs from its admitted V3 unit".into(),
                ));
            }
            if edge.method.role != runmat_types::ClassMethodRole::Ordinary {
                return Err(ArtifactError::Invalid(
                    "graph class method edge does not name an ordinary method".into(),
                ));
            }
            if !seen.insert((edge.caller, edge.method.clone())) {
                return Err(ArtifactError::Invalid(
                    "graph class method edge repeats a constructor method".into(),
                ));
            }
        }
        Ok(())
    }
}
