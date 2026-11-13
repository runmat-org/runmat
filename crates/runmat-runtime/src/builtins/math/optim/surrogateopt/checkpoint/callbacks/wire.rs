//! Callback role metadata surrounds the canonical selected-code JSON owner.
use super::*;
use crate::execution::value_codec::selected_json::{Cooperation, Reader, Writer};
impl ArchivedCallbacks {
    pub(in super::super) async fn write(
        &self,
        writer: &mut Writer,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<()> {
        writer.token(b"{")?;
        writer.key("code", true)?;
        match &self.code {
            Some(code) => writer.portable_code(code, c).await?,
            None => writer.token(b"null")?,
        }
        writer.key("roles", false)?;
        writer.token(b"[")?;
        for (index, role) in self.roles.iter().enumerate() {
            if index != 0 {
                writer.token(b",")?;
            }
            writer.token(b"{")?;
            match role {
                Role::Builtin(value) => {
                    writer.key("Builtin", true)?;
                    writer.string(&value.0, c).await?;
                }
                Role::GraphRoot(value) => {
                    writer.key("GraphRoot", true)?;
                    writer.metadata(value, c).await?;
                }
            }
            writer.token(b"}")?;
        }
        writer.token(b"]")?;
        writer.key("outputs", false)?;
        writer.metadata(&self.outputs, c).await?;
        writer.key("plots", false)?;
        writer.metadata(&self.plots, c).await?;
        writer.token(b"}")
    }
    pub(in super::super) async fn read(
        reader: &mut Reader,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<Self> {
        reader.token(b'{')?;
        reader.key("code", true)?;
        let code = if reader.null()? {
            None
        } else {
            Some(reader.portable_code(ValueLimits::default(), c).await?)
        };
        reader.key("roles", false)?;
        reader.token(b'[')?;
        let mut roles = Vec::new();
        while !reader.is(b']') {
            if !roles.is_empty() {
                reader.token(b',')?;
            }
            reader.token(b'{')?;
            let role = if reader.field_is("Builtin") {
                reader.key("Builtin", true)?;
                Role::Builtin(BuiltinId(reader.string(c).await?))
            } else {
                reader.key("GraphRoot", true)?;
                Role::GraphRoot(reader.metadata(20, c).await?)
            };
            reader.token(b'}')?;
            roles.try_reserve(1).map_err(|source| {
                with_source("checkpoint callback role allocation failed", source)
            })?;
            roles.push(role);
        }
        reader.token(b']')?;
        reader.key("outputs", false)?;
        let outputs = reader.metadata(20, c).await?;
        reader.key("plots", false)?;
        let plots = reader.metadata(20, c).await?;
        reader.token(b'}')?;
        Ok(Self {
            code,
            roles,
            outputs,
            plots,
        })
    }
}
