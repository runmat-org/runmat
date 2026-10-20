//! Admit host-visible controls once. Only genuinely resident controls are
//! materialized and decoded later; data inputs are not touched during admission.
use super::*;

enum TextControl {
    Ready(BuiltinResult<OutputKind>),
    Deferred(Value),
}

impl TextControl {
    fn admit(name: EncodingBuiltin, value: Value, context: &'static str) -> Self {
        if crate::value_contains_gpu(&value) {
            Self::Deferred(value)
        } else {
            Self::Ready(
                scalar_text(&value, name, context).and_then(|text| parse_output_kind(name, &text)),
            )
        }
    }

    fn known_broad_output(&self) -> bool {
        matches!(
            self,
            Self::Ready(Ok(OutputKind::Logical | OutputKind::Cell))
        )
    }

    async fn resolve(
        self,
        name: EncodingBuiltin,
        context: &'static str,
    ) -> BuiltinResult<OutputKind> {
        match self {
            Self::Ready(result) => result,
            Self::Deferred(value) => {
                let value = gather(name, value).await?;
                parse_output_kind(name, &scalar_text(&value, name, context)?)
            }
        }
    }
}

enum ClassControl {
    Ready(BuiltinResult<LabelArray>),
    Deferred(Value),
}

impl ClassControl {
    fn admit(value: Value) -> Self {
        if crate::value_contains_gpu(&value) {
            Self::Deferred(value)
        } else {
            Self::Ready(labels_from_value(EncodingBuiltin::OnehotEncode, value))
        }
    }

    async fn resolve(self) -> BuiltinResult<LabelArray> {
        match self {
            Self::Ready(result) => result,
            Self::Deferred(value) => {
                let value = gather(EncodingBuiltin::OnehotEncode, value).await?;
                labels_from_value(EncodingBuiltin::OnehotEncode, value)
            }
        }
    }
}

enum Step {
    Positional(TextControl),
    Classes(ClassControl),
    NamedOutput(TextControl),
    DeferredName(Value, Value),
    InvalidName(RuntimeError),
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum EncodingFlavor {
    Encode,
    Decode,
}

impl EncodingFlavor {
    const fn name(self) -> EncodingBuiltin {
        match self {
            Self::Encode => EncodingBuiltin::OnehotEncode,
            Self::Decode => EncodingBuiltin::OnehotDecode,
        }
    }

    const fn output_alias_extension(self) -> &'static runmat_builtins::BuiltinExtensionDescriptor {
        match self {
            Self::Encode => &ONEHOTENCODE_OUTPUT_TYPE_ALIAS_EXTENSION,
            Self::Decode => &ONEHOTDECODE_OUTPUT_TYPE_ALIAS_EXTENSION,
        }
    }
}

pub(super) struct AdmittedControls {
    flavor: EncodingFlavor,
    steps: Vec<Step>,
    contains_explicit_gpu: bool,
}

impl AdmittedControls {
    pub(super) fn encode(rest: crate::call::BuiltinArguments) -> Self {
        Self::admit(EncodingFlavor::Encode, rest)
    }

    pub(super) fn decode(rest: crate::call::BuiltinArguments) -> Self {
        Self::admit(EncodingFlavor::Decode, rest)
    }

    fn admit(flavor: EncodingFlavor, rest: crate::call::BuiltinArguments) -> Self {
        let name = flavor.name();
        let allow_classes = flavor == EncodingFlavor::Encode;
        let contains_explicit_gpu = rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu);
        let mut steps = Vec::new();
        let mut values = rest.into_iter();
        let count = values.len();
        if count == 1 || (allow_classes && !count.is_multiple_of(2)) {
            if let Some(value) = values.next() {
                steps.push(Step::Positional(TextControl::admit(
                    name, value, "typename",
                )));
            }
        }
        while let Some(option_name) = values.next() {
            let Some(option_value) = values.next() else {
                steps.push(Step::InvalidName(invalid(
                    name,
                    format!("{name}: name-value options must be paired"),
                )));
                break;
            };
            if crate::value_contains_gpu(&option_name) {
                steps.push(Step::DeferredName(option_name, option_value));
                continue;
            }
            let decoded = scalar_text(&option_name, name, "option name")
                .map(|text| (text.clone(), EncodingOptionName::from_public(&text)));
            match decoded {
                Ok((_, Some(EncodingOptionName::Classes))) if allow_classes => {
                    steps.push(Step::Classes(ClassControl::admit(option_value)));
                }
                Ok((_, Some(EncodingOptionName::OutputType))) => {
                    steps.push(Step::NamedOutput(TextControl::admit(
                        name,
                        option_value,
                        "OutputType",
                    )));
                }
                Ok((text, _)) => steps.push(Step::InvalidName(invalid(
                    name,
                    format!("{name}: unsupported option '{}'", canonical(&text)),
                ))),
                Err(error) => steps.push(Step::InvalidName(error)),
            }
        }
        Self {
            flavor,
            steps,
            contains_explicit_gpu,
        }
    }

    pub(super) fn contains_explicit_gpu(&self) -> bool {
        self.contains_explicit_gpu
    }

    /// Compatibility gates precede all provider access for known controls.
    pub(super) fn preflight_known_extensions(mut self) -> BuiltinResult<Self> {
        if self
            .steps
            .iter()
            .any(|step| matches!(step, Step::NamedOutput(_)))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                self.flavor.output_alias_extension(),
                self.flavor.name().name(),
            )?;
        }
        if self.flavor == EncodingFlavor::Decode
            && self.steps.iter().any(|step| match step {
                Step::Positional(control) | Step::NamedOutput(control) => {
                    control.known_broad_output()
                }
                _ => false,
            })
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ONEHOTDECODE_LOGICAL_CELL_OUTPUT_EXTENSION,
                self.flavor.name().name(),
            )?;
        }
        // Host-visible invalid controls are admitted before resolving any
        // deferred control. Keep the successful typed steps for the ordered
        // execution pass, including repeated options whose last value wins.
        let mut admitted = Vec::with_capacity(self.steps.len());
        for step in self.steps {
            match step {
                Step::Positional(TextControl::Ready(Err(error)))
                | Step::NamedOutput(TextControl::Ready(Err(error)))
                | Step::InvalidName(error) => return Err(error),
                Step::Classes(ClassControl::Ready(Err(error))) => return Err(error),
                step => admitted.push(step),
            }
        }
        self.steps = admitted;
        Ok(self)
    }

    async fn resolve_steps(self) -> BuiltinResult<(Option<LabelArray>, Option<OutputKind>)> {
        let Self { flavor, steps, .. } = self;
        let name = flavor.name();
        let mut classes = None;
        let mut output = None;
        for step in steps {
            match step {
                Step::Positional(control) => {
                    let selected = control.resolve(name, "typename").await?;
                    Self::ensure_broad_output(flavor, selected)?;
                    output = Some(selected);
                }
                Step::Classes(control) => classes = Some(control.resolve().await?),
                Step::NamedOutput(control) => {
                    let selected = control.resolve(name, "OutputType").await?;
                    Self::ensure_broad_output(flavor, selected)?;
                    output = Some(selected);
                }
                Step::DeferredName(option_name, option_value) => {
                    let option_name = gather(name, option_name).await?;
                    let option_name = scalar_text(&option_name, name, "option name")?;
                    match EncodingOptionName::from_public(&option_name) {
                        Some(EncodingOptionName::Classes) if flavor == EncodingFlavor::Encode => {
                            classes = Some(ClassControl::admit(option_value).resolve().await?);
                        }
                        Some(EncodingOptionName::OutputType) => {
                            crate::compatibility::ensure_builtin_extension_enabled(
                                flavor.output_alias_extension(),
                                name.name(),
                            )?;
                            let selected = TextControl::admit(name, option_value, "OutputType")
                                .resolve(name, "OutputType")
                                .await?;
                            Self::ensure_broad_output(flavor, selected)?;
                            output = Some(selected);
                        }
                        _ => {
                            return Err(invalid(
                                name,
                                format!(
                                    "{}: unsupported option '{}'",
                                    name,
                                    canonical(&option_name)
                                ),
                            ))
                        }
                    }
                }
                Step::InvalidName(error) => return Err(error),
            }
        }
        Ok((classes, output))
    }

    fn ensure_broad_output(flavor: EncodingFlavor, selected: OutputKind) -> BuiltinResult<()> {
        if flavor == EncodingFlavor::Decode
            && matches!(selected, OutputKind::Logical | OutputKind::Cell)
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ONEHOTDECODE_LOGICAL_CELL_OUTPUT_EXTENSION,
                flavor.name().name(),
            )?;
        }
        Ok(())
    }

    pub(super) async fn resolve_encode(self) -> BuiltinResult<OnehotEncodeOptions> {
        let (classes, output) = self.resolve_steps().await?;
        Ok(OnehotEncodeOptions { classes, output })
    }

    pub(super) async fn resolve_decode(self) -> BuiltinResult<OnehotDecodeOptions> {
        let (_, output) = self.resolve_steps().await?;
        Ok(OnehotDecodeOptions { output })
    }
}
