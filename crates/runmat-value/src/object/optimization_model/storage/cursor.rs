//! Fixed-depth metadata walk; each advance handles one physical owner/slot.
use super::backing::invalid;
use super::*;
use crate::optimization_variable::OptimizationVariablePayload as Variable;

#[derive(PartialEq)]
enum Pass {
    Quote,
    Quoted,
    Admit,
    Done,
}
pub struct OptimizationNativeAdmission<S: OptimizationNativeSource = OptimizationNativeInput> {
    input: S,
    phase: u8,
    axis: usize,
    label: Option<usize>,
    extra: usize,
    pass: Pass,
}
impl<S: OptimizationNativeSource> OptimizationNativeAdmission<S> {
    pub(super) fn new(input: S) -> Self {
        Self {
            input,
            phase: 0,
            axis: 0,
            label: None,
            extra: 0,
            pass: Pass::Quote,
        }
    }
    pub fn input(&self) -> &S {
        &self.input
    }
    pub fn quote_next(
        &mut self,
    ) -> Result<Option<OptimizationMetadataQuote>, ExactArrayClaimPromotionError> {
        if self.pass != Pass::Quote {
            return Err(invalid());
        }
        match self.next()? {
            Some(value) => value.quote().map(Some),
            None => {
                self.pass = Pass::Quoted;
                Ok(None)
            }
        }
    }
    pub fn begin_admission(&mut self) -> Result<(), ExactArrayClaimPromotionError> {
        if self.pass != Pass::Quoted {
            return Err(invalid());
        }
        self.phase = 0;
        self.axis = 0;
        self.label = None;
        self.extra = 0;
        self.pass = Pass::Admit;
        Ok(())
    }
    pub fn advance_admission(
        &mut self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<bool, ExactArrayClaimPromotionError> {
        if self.pass != Pass::Admit {
            return Err(invalid());
        }
        match self.next()? {
            Some(value) => {
                value.promote(permit)?;
                Ok(true)
            }
            None => {
                self.pass = Pass::Done;
                Ok(false)
            }
        }
    }
    fn next(&mut self) -> Result<Option<MetadataRef<'_>>, ExactArrayClaimPromotionError> {
        if self.input.variable().is_none() {
            return Err(invalid());
        }
        let value = match self.phase {
            0 => {
                self.phase = 1;
                match fields(&self.input).0 {
                    Some(value) => MetadataRef::new(value),
                    None => MetadataRef::empty(),
                }
            }
            1 => {
                self.phase = 2;
                match fields(&self.input).1 {
                    Some(value) => MetadataRef::new(value),
                    None => MetadataRef::empty(),
                }
            }
            2 => {
                self.phase = 3;
                match fields(&self.input).2 {
                    Some(value) => MetadataRef::new(value),
                    None => MetadataRef::empty(),
                }
            }
            3 => {
                let names = fields(&self.input).2;
                match names.and_then(|names| names.get(self.axis)) {
                    None => {
                        self.phase = 4;
                        MetadataRef::empty()
                    }
                    Some(None) => {
                        self.axis += 1;
                        MetadataRef::empty()
                    }
                    Some(Some(labels)) => match self.label {
                        None => {
                            self.label = Some(0);
                            MetadataRef::new(labels)
                        }
                        Some(index) => match labels.get(index) {
                            Some(value) => {
                                self.label = Some(index + 1);
                                MetadataRef::new(value)
                            }
                            None => {
                                self.axis += 1;
                                self.label = None;
                                MetadataRef::empty()
                            }
                        },
                    },
                }
            }
            4 => {
                let item = extra(&self.input, self.extra);
                self.extra = self.extra.checked_add(1).ok_or_else(invalid)?;
                match item {
                    Some(value) => value,
                    None => {
                        self.phase = 5;
                        MetadataRef::empty()
                    }
                }
            }
            5 => {
                self.phase = 6;
                self.input
                    .existing_object()
                    .and_then(ObjectInstance::optimization_metadata_header)
                    .unwrap_or_else(MetadataRef::empty)
            }
            6 => {
                self.phase = 7;
                self.input
                    .existing_object()
                    .map(|object| MetadataRef::new(object.class_name_owner()))
                    .unwrap_or_else(MetadataRef::empty)
            }
            _ => return Ok(None),
        };
        Ok(Some(value))
    }
}
fn fields<S: OptimizationNativeSource>(
    input: &S,
) -> (
    Option<&crate::Utf8Text>,
    Option<&crate::SharedShape>,
    Option<&crate::optimization_model::ModelAxisNames>,
) {
    match input.variable() {
        Some(Variable::Definition(data)) => (
            Some(data.name_owner()),
            Some(data.shape()),
            Some(data.index_names()),
        ),
        Some(Variable::Reference(data)) => (None, Some(data.shape()), None),
        None => (None, None, None),
    }
}
fn extra<S: OptimizationNativeSource>(input: &S, index: usize) -> Option<MetadataRef<'_>> {
    match input.variable()? {
        Variable::Reference(data) => match index {
            0 => Some(MetadataRef::new(data.axis_sources())),
            1 => data.index_storage().map(MetadataRef::new),
            _ => None,
        },
        Variable::Definition(_) => None,
    }
}

impl OptimizationNativeAdmission<OptimizationNativeInput> {
    pub fn finish(self) -> Result<PreparedOptimizationNative, ExactArrayClaimPromotionError> {
        if self.pass == Pass::Done {
            Ok(PreparedOptimizationNative(self.input))
        } else {
            Err(invalid())
        }
    }
}
