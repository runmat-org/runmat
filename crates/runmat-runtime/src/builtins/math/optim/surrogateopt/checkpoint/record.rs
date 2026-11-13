//! Private versioned payload; portable code bytes are data until canonical
//! artifact/source/static/capture admission succeeds in the current caller.
use super::super::{
    campaign,
    initial::{ArchivedInitial, BorrowedInitial},
    notice::Frame,
};
use super::{geometry, options::SavedOptions, trials, ArchivedCallbacks};
use crate::builtins::stats::random::rng::PortableSnapshot;
use runmat_execution::ProgramEnvironment;

pub(super) const VERSION: u16 = 2;
pub(super) struct Borrowed<'a> {
    pub version: u16,
    pub environment: ProgramEnvironment,
    pub options: &'a SavedOptions,
    pub geometry: geometry::Borrowed<'a>,
    pub trials: trials::Borrowed<'a>,
    pub initial: Option<BorrowedInitial<'a>>,
    pub initial_rng: &'a PortableSnapshot,
    pub current_rng: PortableSnapshot,
    pub frame: Frame,
    pub campaign: campaign::State,
    pub notices: &'a [Frame],
    pub callbacks: ArchivedCallbacks,
}
impl Borrowed<'_> {
    pub(super) async fn encode(
        self,
        c: &mut super::json::Cooperation<'_>,
    ) -> crate::BuiltinResult<Vec<u8>> {
        let mut writer = super::json::Writer::new();
        writer.token(b"{")?;
        writer.key("version", true)?;
        writer.metadata(&self.version, c).await?;
        writer.key("environment", false)?;
        writer.metadata(&self.environment, c).await?;
        writer.key("options", false)?;
        self.options.write(&mut writer, c).await?;
        writer.key("geometry", false)?;
        self.geometry.write(&mut writer, c).await?;
        writer.key("trials", false)?;
        self.trials.write(&mut writer, c).await?;
        writer.key("initial", false)?;
        match self.initial {
            Some(initial) => initial.write(&mut writer, c).await?,
            None => writer.token(b"null")?,
        }
        writer.key("initial_rng", false)?;
        writer.metadata(self.initial_rng, c).await?;
        writer.key("current_rng", false)?;
        writer.metadata(&self.current_rng, c).await?;
        writer.key("frame", false)?;
        writer.metadata(&self.frame, c).await?;
        writer.key("campaign", false)?;
        writer.metadata(&self.campaign, c).await?;
        writer.key("notices", false)?;
        writer.array(self.notices, c).await?;
        writer.key("callbacks", false)?;
        self.callbacks.write(&mut writer, c).await?;
        writer.token(b"}")?;
        Ok(writer.finish())
    }
}

pub(super) struct Saved {
    pub version: u16,
    pub environment: ProgramEnvironment,
    pub options: SavedOptions,
    pub geometry: geometry::Geometry,
    pub trials: trials::Trials,
    pub initial: Option<ArchivedInitial>,
    pub initial_rng: PortableSnapshot,
    pub current_rng: PortableSnapshot,
    pub frame: Frame,
    pub campaign: campaign::State,
    pub notices: Vec<Frame>,
    pub callbacks: ArchivedCallbacks,
}
impl Saved {
    pub(super) async fn decode(
        bytes: Vec<u8>,
        expected: &ProgramEnvironment,
        c: &mut super::json::Cooperation<'_>,
    ) -> crate::BuiltinResult<Self> {
        let mut reader = super::json::Reader::new(bytes);
        reader.token(b'{')?;
        reader.key("version", true)?;
        let version = reader.metadata(5, c).await?;
        reader.key("environment", false)?;
        let environment: ProgramEnvironment = reader.metadata(2048, c).await?;
        if version != VERSION || &environment != expected {
            return Err(super::error(
                "checkpoint version or execution environment is incompatible",
            ));
        }
        reader.key("options", false)?;
        let options = SavedOptions::read(&mut reader, c).await?;
        reader.key("geometry", false)?;
        let geometry = geometry::Geometry::read(&mut reader, c).await?;
        reader.key("trials", false)?;
        let trials = trials::Trials::read(&mut reader, c).await?;
        reader.key("initial", false)?;
        let initial = if reader.null()? {
            None
        } else {
            Some(ArchivedInitial::read(&mut reader, c).await?)
        };
        // Public RNG state has at most 625 U32 words. This bound includes all
        // canonical decimal words and its fixed algorithm/seed field schema.
        const RNG_JSON: usize = 625 * 11 + 1024;
        reader.key("initial_rng", false)?;
        let initial_rng = reader.metadata(RNG_JSON, c).await?;
        reader.key("current_rng", false)?;
        let current_rng = reader.metadata(RNG_JSON, c).await?;
        reader.key("frame", false)?;
        let frame = reader.metadata(2048, c).await?;
        reader.key("campaign", false)?;
        let campaign = reader.metadata(1024, c).await?;
        reader.key("notices", false)?;
        let notices = reader.array(2048, c).await?;
        reader.key("callbacks", false)?;
        let callbacks = ArchivedCallbacks::read(&mut reader, c).await?;
        reader.token(b'}')?;
        reader.finish()?;
        Ok(Self {
            version,
            environment,
            options,
            geometry,
            trials,
            initial,
            initial_rng,
            current_rng,
            frame,
            campaign,
            notices,
            callbacks,
        })
    }
}
