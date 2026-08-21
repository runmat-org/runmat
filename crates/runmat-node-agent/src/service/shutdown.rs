#[derive(Debug)]
pub struct Shutdown {
    sender: tokio::sync::watch::Sender<bool>,
    receiver: tokio::sync::watch::Receiver<bool>,
}

impl Default for Shutdown {
    fn default() -> Self {
        let (sender, receiver) = tokio::sync::watch::channel(false);
        Self { sender, receiver }
    }
}

impl Shutdown {
    pub fn trigger(&self) {
        self.sender.send_replace(true);
    }

    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<bool> {
        self.receiver.clone()
    }
}
