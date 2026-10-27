//! Existing PortableSelectedCode field schema, streamed into one byte owner.
//! Raw code/archive bytes never become execution authority in this layer.
use super::*;
use runmat_execution::value::{GraphClassMethodDependency, PortableSelectedCode, ValueLimits};
impl Writer {
    pub(crate) async fn portable_code(
        &mut self,
        value: &PortableSelectedCode,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<()> {
        self.token(b"{")?;
        self.key("schema_version", true)?;
        self.metadata(&value.schema_version, c).await?;
        self.key("primary_unit", false)?;
        self.array(&value.primary_unit, c).await?;
        self.key("retained_units", false)?;
        self.token(b"[")?;
        for (index, unit) in value.retained_units.iter().enumerate() {
            if index != 0 {
                self.token(b",")?;
            }
            self.array(unit, c).await?;
        }
        self.token(b"]")?;
        self.key("graph", false)?;
        self.graph(&value.graph, c).await?;
        self.key("class_dependencies", false)?;
        self.array(&value.class_dependencies, c).await?;
        self.key("class_method_dependencies", false)?;
        self.token(b"[")?;
        for (index, dependency) in value.class_method_dependencies.iter().enumerate() {
            if index != 0 {
                self.token(b",")?;
            }
            self.token(b"{")?;
            self.key("caller", true)?;
            self.metadata(&dependency.caller, c).await?;
            self.key("method", false)?;
            self.token(b"{")?;
            self.key("declaring_class", true)?;
            self.string(dependency.method.declaring_class.display_name(), c)
                .await?;
            self.key("method", false)?;
            self.string(dependency.method.method.display_name(), c)
                .await?;
            self.key("is_static", false)?;
            self.metadata(&dependency.method.is_static, c).await?;
            self.key("role", false)?;
            self.metadata(&dependency.method.role, c).await?;
            self.token(b"}")?;
            self.key("target", false)?;
            self.metadata(&dependency.target, c).await?;
            self.key("target_unit_digest", false)?;
            self.metadata(&dependency.target_unit_digest, c).await?;
            self.key("target_body_digest", false)?;
            self.metadata(&dependency.target_body_digest, c).await?;
            self.token(b"}")?;
        }
        self.token(b"]")?;
        self.key("static_dependencies", false)?;
        self.array(&value.static_dependencies, c).await?;
        self.key("selected_bundle_archives", false)?;
        self.token(b"[")?;
        for (index, archive) in value.selected_bundle_archives.iter().enumerate() {
            if index != 0 {
                self.token(b",")?;
            }
            match archive {
                Some(bytes) => self.array(bytes, c).await?,
                None => self.token(b"null")?,
            }
        }
        self.token(b"]")?;
        self.key("selected_foreign_policies", false)?;
        self.array(&value.selected_foreign_policies, c).await?;
        self.token(b"}")
    }
}
impl Reader {
    pub(crate) async fn portable_code(
        &mut self,
        limits: ValueLimits,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<PortableSelectedCode> {
        self.token(b'{')?;
        self.key("schema_version", true)?;
        let schema_version = self.metadata(5, c).await?;
        self.key("primary_unit", false)?;
        let primary_unit = self.array(3, c).await?;
        self.key("retained_units", false)?;
        self.token(b'[')?;
        let mut retained_units = Vec::new();
        while !self.is(b']') {
            if retained_units.len() >= runmat_execution::value::GRAPH_MAX_SELECTED_UNITS - 1 {
                return Err(error(
                    "selected JSON exceeds the graph unit coordinate domain",
                ));
            }
            if !retained_units.is_empty() {
                self.token(b',')?;
            }
            retained_units.try_reserve(1).map_err(|source| {
                with_source("selected code-unit index allocation failed", source)
            })?;
            retained_units.push(self.array(3, c).await?);
        }
        self.token(b']')?;
        self.key("graph", false)?;
        let graph = self.graph(limits, c).await?;
        self.key("class_dependencies", false)?;
        let class_dependencies = self.array(512, c).await?;
        self.key("class_method_dependencies", false)?;
        self.token(b'[')?;
        let mut class_method_dependencies = Vec::new();
        while !self.is(b']') {
            if !class_method_dependencies.is_empty() {
                self.token(b',')?;
            }
            let value = method(self, c).await?;
            class_method_dependencies
                .try_reserve(1)
                .map_err(|source| with_source("selected method-edge allocation failed", source))?;
            class_method_dependencies.push(value);
        }
        self.token(b']')?;
        self.key("static_dependencies", false)?;
        let static_dependencies = self.array(1024, c).await?;
        self.key("selected_bundle_archives", false)?;
        self.token(b'[')?;
        let mut selected_bundle_archives = Vec::new();
        while !self.is(b']') {
            if selected_bundle_archives.len() >= runmat_execution::value::GRAPH_MAX_SELECTED_UNITS {
                return Err(error(
                    "selected archive index exceeds the graph unit domain",
                ));
            }
            if !selected_bundle_archives.is_empty() {
                self.token(b',')?;
            }
            selected_bundle_archives.try_reserve(1).map_err(|source| {
                with_source("selected archive index allocation failed", source)
            })?;
            selected_bundle_archives.push(if self.null()? {
                None
            } else {
                Some(self.array(3, c).await?)
            });
        }
        self.token(b']')?;
        self.key("selected_foreign_policies", false)?;
        let selected_foreign_policies = self
            .array_limited(256, runmat_execution::value::GRAPH_MAX_SELECTED_UNITS, c)
            .await?;
        self.token(b'}')?;
        Ok(PortableSelectedCode {
            schema_version,
            primary_unit,
            retained_units,
            graph,
            class_dependencies,
            class_method_dependencies,
            static_dependencies,
            selected_bundle_archives,
            selected_foreign_policies,
        })
    }
}
async fn method(
    reader: &mut Reader,
    c: &mut Cooperation<'_>,
) -> BuiltinResult<GraphClassMethodDependency> {
    reader.token(b'{')?;
    reader.key("caller", true)?;
    let caller = reader.metadata(256, c).await?;
    reader.key("method", false)?;
    reader.token(b'{')?;
    reader.key("declaring_class", true)?;
    let declaring_class = runmat_types::ClassIdentity::new(reader.string(c).await?)
        .map_err(|source| with_source("invalid selected declaring class", source))?;
    reader.key("method", false)?;
    let name = runmat_types::MethodName(reader.string(c).await?);
    reader.key("is_static", false)?;
    let is_static = reader.metadata(5, c).await?;
    reader.key("role", false)?;
    let role = reader.metadata(64, c).await?;
    reader.token(b'}')?;
    reader.key("target", false)?;
    let target = reader.metadata(256, c).await?;
    reader.key("target_unit_digest", false)?;
    let target_unit_digest = reader.metadata(128, c).await?;
    reader.key("target_body_digest", false)?;
    let target_body_digest = reader.metadata(128, c).await?;
    reader.token(b'}')?;
    Ok(GraphClassMethodDependency {
        caller,
        method: runmat_types::ClassMethodOwner {
            declaring_class,
            method: name,
            is_static,
            role,
        },
        target,
        target_unit_digest,
        target_body_digest,
    })
}
