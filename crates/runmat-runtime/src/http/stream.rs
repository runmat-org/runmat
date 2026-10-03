use futures::stream::{LocalBoxStream, StreamExt};

use super::TransportError;

/// Response metadata is available before the caller opens a partial output.
#[derive(Debug)]
pub struct HttpResponseHead {
    pub headers: Vec<(String, String)>,
    pub content_length: Option<u64>,
}

/// One owner of an HTTP response body. Dropping this value drops its native
/// abort guard; a body can be consumed only in order and only once.
pub struct HttpBodyStream {
    pub head: HttpResponseHead,
    body: LocalBoxStream<'static, Result<Vec<u8>, TransportError>>,
}

impl HttpBodyStream {
    pub(crate) fn new(
        head: HttpResponseHead,
        body: LocalBoxStream<'static, Result<Vec<u8>, TransportError>>,
    ) -> Self {
        Self { head, body }
    }

    pub async fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, TransportError> {
        self.body.next().await.transpose()
    }

    pub(crate) fn abort(&mut self) {
        self.body = Box::pin(futures::stream::empty());
    }
}
