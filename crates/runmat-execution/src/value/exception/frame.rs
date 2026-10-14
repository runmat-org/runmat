use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExceptionCallFrame {
    pub function: String,
    pub source: ExceptionFrameSource,
    pub span: Option<(u64, u64)>,
}

/// Detached presentation never authenticates source code or a callable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExceptionFrameSource {
    Unknown,
    Admitted(runmat_types::ProgramSourceId),
    Detached(DetachedDiagnosticFrame),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetachedDiagnosticFrame {
    pub display_name: String,
    pub line: Option<u64>,
    pub column: Option<u64>,
}

impl ExceptionCallFrame {
    pub fn program_source_id(&self) -> Option<runmat_types::ProgramSourceId> {
        match &self.source {
            ExceptionFrameSource::Admitted(id) => Some(*id),
            ExceptionFrameSource::Unknown | ExceptionFrameSource::Detached(_) => None,
        }
    }
}

/// Private decode adapter retains Direct V7's `function, source_id, span`
/// field shape. Serialization borrows those fields and adds presentation only
/// for the typed Detached variant.
#[derive(Deserialize)]
struct ExceptionCallFrameWire {
    function: String,
    #[serde(rename = "source_id")]
    program_source_id: Option<runmat_types::ProgramSourceId>,
    span: Option<(u64, u64)>,
    #[serde(default)]
    detached_presentation: Option<DetachedDiagnosticFrame>,
}

impl Serialize for ExceptionCallFrame {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct(
            "ExceptionCallFrame",
            if matches!(&self.source, ExceptionFrameSource::Detached(_)) {
                4
            } else {
                3
            },
        )?;
        state.serialize_field("function", &self.function)?;
        state.serialize_field("source_id", &self.program_source_id())?;
        state.serialize_field("span", &self.span)?;
        if let ExceptionFrameSource::Detached(presentation) = &self.source {
            state.serialize_field("detached_presentation", presentation)?;
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for ExceptionCallFrame {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ExceptionCallFrameWire::deserialize(deserializer)?;
        let source = match (wire.program_source_id, wire.detached_presentation) {
            (Some(id), None) => ExceptionFrameSource::Admitted(id),
            (None, Some(presentation))
                if presentation.line.is_some() == presentation.column.is_some() =>
            {
                ExceptionFrameSource::Detached(presentation)
            }
            (None, Some(_)) => {
                return Err(serde::de::Error::custom(
                    "detached exception coordinates must be present together",
                ));
            }
            (None, None) => ExceptionFrameSource::Unknown,
            (Some(_), Some(_)) => {
                return Err(serde::de::Error::custom(
                    "exception frame cannot claim admitted and detached sources",
                ));
            }
        };
        Ok(Self {
            function: wire.function,
            source,
            span: wire.span,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{
        ExceptionFailureOrigin, ExceptionGpuGatherRetry, ExceptionRuntimeMetadata, ExceptionValue,
        InlineValue, ValueLimits, ValuePayload,
    };

    #[test]
    fn direct_frame_json_is_unchanged_and_conflicting_sources_are_rejected() {
        let admitted = ExceptionCallFrame {
            function: "origin".into(),
            source: ExceptionFrameSource::Admitted(runmat_types::ProgramSourceId(7)),
            span: Some((2, 5)),
        };
        assert_eq!(
            serde_json::to_string(&admitted).unwrap(),
            r#"{"function":"origin","source_id":7,"span":[2,5]}"#
        );
        let unknown = ExceptionCallFrame {
            function: "origin".into(),
            source: ExceptionFrameSource::Unknown,
            span: None,
        };
        assert_eq!(
            serde_json::to_string(&unknown).unwrap(),
            r#"{"function":"origin","source_id":null,"span":null}"#
        );
        assert!(serde_json::from_str::<ExceptionCallFrame>(
            r#"{"function":"origin","source_id":7,"span":[2,5],"detached_presentation":{"display_name":"origin.m","line":1,"column":3}}"#
        )
        .is_err());
    }

    #[test]
    fn detached_frame_is_presentation_only_and_not_a_direct_v7_source() {
        let detached = ExceptionCallFrame {
            function: "origin".into(),
            source: ExceptionFrameSource::Detached(DetachedDiagnosticFrame {
                display_name: "origin.m".into(),
                line: Some(2),
                column: Some(4),
            }),
            span: Some((3, 1)),
        };
        let wire = serde_json::to_string(&detached).unwrap();
        assert!(wire.contains("detached_presentation"));
        assert_eq!(
            serde_json::from_str::<ExceptionCallFrame>(&wire).unwrap(),
            detached
        );
        let spanless = ExceptionCallFrame {
            function: "origin".into(),
            source: ExceptionFrameSource::Detached(DetachedDiagnosticFrame {
                display_name: "origin.m".into(),
                line: None,
                column: None,
            }),
            span: None,
        };
        let wire = serde_json::to_string(&spanless).unwrap();
        assert_eq!(
            serde_json::from_str::<ExceptionCallFrame>(&wire).unwrap(),
            spanless
        );
        assert!(serde_json::from_str::<ExceptionCallFrame>(
            r#"{"function":"origin","source_id":null,"span":null,"detached_presentation":{"display_name":"origin.m","line":1,"column":null}}"#
        )
        .is_err());
        let value = ValuePayload::Inline(Box::new(InlineValue::Exception(ExceptionValue {
            identifier: "RunMat:diagnostic".into(),
            message: "failure".into(),
            stack: Vec::new(),
            causes: Vec::new(),
            runtime: Some(ExceptionRuntimeMetadata {
                span: None,
                builtin: None,
                task_id: None,
                call_frames: vec![detached],
                call_frames_elided: 0,
                call_stack: Vec::new(),
                phase: None,
                gpu_gather_retry: ExceptionGpuGatherRetry::Legacy,
                failure_origin: ExceptionFailureOrigin::Unclassified,
            }),
        })));
        assert!(value.validate(ValueLimits::default()).is_ok());
        assert!(value
            .validate_for_program_sources(ValueLimits::default(), None)
            .is_err());
        assert!(value
            .validate_for_graph_sources(ValueLimits::default(), None)
            .is_ok());
    }
}
