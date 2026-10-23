use std::{
    io::{self, Read, Write},
    net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};

// The application and three semantic controls each have a 30-second process
// limit. Keep one bounded owner deadline across that sequence plus startup.
const OVERALL_TIMEOUT: Duration = Duration::from_secs(150);
const IO_SLICE: Duration = Duration::from_millis(200);
const ACCEPT_POLL: Duration = Duration::from_millis(10);

pub struct LoopbackWitness {
    results: Vec<mpsc::Receiver<Result<()>>>,
    threads: Vec<thread::JoinHandle<()>>,
    cancelled: Arc<AtomicBool>,
    deadline: Instant,
}

impl LoopbackWitness {
    pub fn start(http_port: u16, tcp_port: u16, calls: usize) -> Result<Self> {
        let http = bind(http_port, "HTTP")?;
        let tcp = bind(tcp_port, "TCP")?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let deadline = Instant::now() + OVERALL_TIMEOUT;
        let (http_tx, http_rx) = mpsc::sync_channel(1);
        let (tcp_tx, tcp_rx) = mpsc::sync_channel(1);
        let http_cancelled = Arc::clone(&cancelled);
        let http_thread = thread::spawn(move || {
            let _ = http_tx.send(serve_http(http, calls, deadline, &http_cancelled));
        });
        let tcp_cancelled = Arc::clone(&cancelled);
        let tcp_thread = thread::spawn(move || {
            let _ = tcp_tx.send(serve_tcp(tcp, calls, deadline, &tcp_cancelled));
        });
        Ok(Self {
            results: vec![http_rx, tcp_rx],
            threads: vec![http_thread, tcp_thread],
            cancelled,
            deadline,
        })
    }

    pub fn finish(mut self) -> Result<()> {
        let mut failure = None;
        for result in self.results.drain(..) {
            let remaining = self.deadline.saturating_duration_since(Instant::now());
            let received = if remaining.is_zero() {
                Err(mpsc::RecvTimeoutError::Timeout)
            } else {
                result.recv_timeout(remaining)
            };
            match received {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    failure.get_or_insert(error);
                    self.cancelled.store(true, Ordering::Release);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    failure.get_or_insert_with(|| {
                        anyhow::anyhow!("loopback witness exceeded its overall deadline")
                    });
                    self.cancelled.store(true, Ordering::Release);
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    failure.get_or_insert_with(|| {
                        anyhow::anyhow!("loopback witness exited without a result")
                    });
                    self.cancelled.store(true, Ordering::Release);
                }
            }
        }
        if let Err(error) = self.stop_and_join() {
            failure.get_or_insert(error);
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn stop_and_join(&mut self) -> Result<()> {
        self.cancelled.store(true, Ordering::Release);
        let mut failure = None;
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                failure.get_or_insert_with(|| anyhow::anyhow!("loopback witness panicked"));
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl Drop for LoopbackWitness {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

fn bind(port: u16, kind: &str) -> Result<TcpListener> {
    let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port))
        .with_context(|| format!("bind fixed {kind} loopback port {port}"))?;
    listener
        .set_nonblocking(true)
        .context("configure nonblocking loopback listener")?;
    Ok(listener)
}

fn serve_http(
    listener: TcpListener,
    calls: usize,
    deadline: Instant,
    cancelled: &AtomicBool,
) -> Result<()> {
    const MAX_REQUEST: usize = 16 * 1024;
    for _ in 0..calls {
        let (mut stream, peer) = accept_until(&listener, deadline, cancelled, "HTTP")?;
        if !peer.ip().is_loopback() {
            bail!("HTTP witness accepted a non-loopback peer");
        }
        configure_stream(&stream)?;
        let mut request = Vec::new();
        let mut chunk = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let count = read_until(&mut stream, &mut chunk, deadline, cancelled, "HTTP request")?;
            if count == 0 || request.len().saturating_add(count) > MAX_REQUEST {
                bail!("HTTP witness request ended early or exceeded its bound");
            }
            request.extend_from_slice(&chunk[..count]);
        }
        let request = std::str::from_utf8(&request).context("HTTP request is not UTF-8")?;
        if !request.starts_with("GET /witness ") || !request.contains("\r\n\r\n") {
            bail!("HTTP witness received an unexpected bounded request");
        }
        let body = b"runmat-http";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        write_all_until(
            &mut stream,
            response.as_bytes(),
            deadline,
            cancelled,
            "HTTP response header",
        )?;
        write_all_until(&mut stream, body, deadline, cancelled, "HTTP response body")?;
        flush_until(&mut stream, deadline, cancelled, "HTTP response")?;
    }
    Ok(())
}

fn serve_tcp(
    listener: TcpListener,
    calls: usize,
    deadline: Instant,
    cancelled: &AtomicBool,
) -> Result<()> {
    for _ in 0..calls {
        let (mut stream, peer) = accept_until(&listener, deadline, cancelled, "TCP")?;
        if !peer.ip().is_loopback() {
            bail!("TCP witness accepted a non-loopback peer");
        }
        configure_stream(&stream)?;
        let mut request = [0_u8; 1];
        read_exact_until(
            &mut stream,
            &mut request,
            deadline,
            cancelled,
            "TCP witness byte",
        )?;
        if request != [9] {
            bail!("TCP witness received the wrong request byte");
        }
        write_all_until(
            &mut stream,
            &[1, 2, 3],
            deadline,
            cancelled,
            "TCP witness reply",
        )?;
        flush_until(&mut stream, deadline, cancelled, "TCP witness reply")?;
    }
    Ok(())
}

fn accept_until(
    listener: &TcpListener,
    deadline: Instant,
    cancelled: &AtomicBool,
    kind: &str,
) -> Result<(TcpStream, std::net::SocketAddr)> {
    loop {
        check_live(deadline, cancelled, kind)?;
        match listener.accept() {
            Ok(connection) => return Ok(connection),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(ACCEPT_POLL.min(deadline.saturating_duration_since(Instant::now())));
            }
            Err(error) => return Err(error).with_context(|| format!("accept {kind} witness")),
        }
    }
}

fn configure_stream(stream: &TcpStream) -> Result<()> {
    // Accepted sockets do not inherit the listener's nonblocking state
    // uniformly across platforms. Use bounded blocking I/O on each stream.
    stream
        .set_nonblocking(false)
        .context("configure blocking loopback stream")?;
    stream
        .set_read_timeout(Some(IO_SLICE))
        .context("configure loopback read timeout")?;
    stream
        .set_write_timeout(Some(IO_SLICE))
        .context("configure loopback write timeout")?;
    Ok(())
}

fn read_until(
    stream: &mut TcpStream,
    buffer: &mut [u8],
    deadline: Instant,
    cancelled: &AtomicBool,
    operation: &str,
) -> Result<usize> {
    loop {
        check_live(deadline, cancelled, operation)?;
        match stream.read(buffer) {
            Ok(count) => return Ok(count),
            Err(error) if retryable(&error) => {}
            Err(error) => return Err(error).with_context(|| format!("read {operation}")),
        }
    }
}

fn read_exact_until(
    stream: &mut TcpStream,
    mut buffer: &mut [u8],
    deadline: Instant,
    cancelled: &AtomicBool,
    operation: &str,
) -> Result<()> {
    while !buffer.is_empty() {
        let count = read_until(stream, buffer, deadline, cancelled, operation)?;
        if count == 0 {
            bail!("{operation} ended early");
        }
        buffer = &mut buffer[count..];
    }
    Ok(())
}

fn write_all_until(
    stream: &mut TcpStream,
    mut bytes: &[u8],
    deadline: Instant,
    cancelled: &AtomicBool,
    operation: &str,
) -> Result<()> {
    while !bytes.is_empty() {
        check_live(deadline, cancelled, operation)?;
        match stream.write(bytes) {
            Ok(0) => bail!("{operation} made no progress"),
            Ok(count) => bytes = &bytes[count..],
            Err(error) if retryable(&error) => {}
            Err(error) => return Err(error).with_context(|| format!("write {operation}")),
        }
    }
    Ok(())
}

fn flush_until(
    stream: &mut TcpStream,
    deadline: Instant,
    cancelled: &AtomicBool,
    operation: &str,
) -> Result<()> {
    loop {
        check_live(deadline, cancelled, operation)?;
        match stream.flush() {
            Ok(()) => return Ok(()),
            Err(error) if retryable(&error) => {}
            Err(error) => return Err(error).with_context(|| format!("flush {operation}")),
        }
    }
}

fn check_live(deadline: Instant, cancelled: &AtomicBool, operation: &str) -> Result<()> {
    if cancelled.load(Ordering::Acquire) {
        bail!("{operation} cancelled");
    }
    if Instant::now() >= deadline {
        bail!("{operation} exceeded the loopback deadline");
    }
    Ok(())
}

fn retryable(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted
    )
}
