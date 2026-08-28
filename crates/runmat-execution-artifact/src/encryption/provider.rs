use super::{EncryptedArtifact, EncryptionContext, ExecutionRecipientKey};
use crate::ArtifactResult;

pub trait ExecutionEncryptionProvider {
    type PrivateKey;

    fn seal(
        &self,
        recipient: &ExecutionRecipientKey,
        context: EncryptionContext,
        plaintext: &[u8],
    ) -> ArtifactResult<EncryptedArtifact>;

    fn open(
        &self,
        private_key: &Self::PrivateKey,
        artifact: &EncryptedArtifact,
    ) -> ArtifactResult<Vec<u8>>;
}
