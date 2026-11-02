//! Rank communication: neighbor exchanges and small collective reductions.
//!
//! Reductions sum contributions in rank order, so every rank obtains bitwise identical values
//! and a run is reproducible for a fixed rank count.

use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver, Sender};

use crate::error::{DistributedError, DistributedResult};

pub trait Communicator: Send {
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    /// Send one buffer to each listed peer and receive one buffer from each, in `peers` order.
    fn exchange(
        &mut self,
        peers: &[usize],
        sends: Vec<Vec<f64>>,
    ) -> DistributedResult<Vec<Vec<f64>>>;
    /// Contribution of every rank, indexed by rank.
    fn allgather(&mut self, local: &[f64]) -> DistributedResult<Vec<Vec<f64>>>;
    /// Elementwise sum over ranks, accumulated in rank order.
    fn allreduce_sum(&mut self, values: &mut [f64]) -> DistributedResult<()> {
        let contributions = self.allgather(values)?;
        values.iter_mut().for_each(|value| *value = 0.0);
        for contribution in contributions {
            if contribution.len() != values.len() {
                return Err(DistributedError::Communication(
                    "reduction contributions differ in length".into(),
                ));
            }
            for (value, part) in values.iter_mut().zip(contribution) {
                *value += part;
            }
        }
        Ok(())
    }
}

/// Ranks running as threads of one process.
pub struct ThreadCommunicator {
    rank: usize,
    size: usize,
    senders: Vec<Option<Sender<Vec<f64>>>>,
    receivers: Vec<Option<Receiver<Vec<f64>>>>,
}

impl ThreadCommunicator {
    pub fn group(size: usize) -> Vec<Self> {
        let mut senders = (0..size)
            .map(|_| (0..size).map(|_| None).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut receivers = (0..size)
            .map(|_| (0..size).map(|_| None).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for (from, outbound) in senders.iter_mut().enumerate() {
            for (to, inbound) in receivers.iter_mut().enumerate() {
                if from != to {
                    let (sender, receiver) = channel();
                    outbound[to] = Some(sender);
                    inbound[from] = Some(receiver);
                }
            }
        }
        senders
            .into_iter()
            .zip(receivers)
            .enumerate()
            .map(|(rank, (senders, receivers))| Self {
                rank,
                size,
                senders,
                receivers,
            })
            .collect()
    }

    fn send(&self, peer: usize, values: Vec<f64>) -> DistributedResult<()> {
        self.senders[peer]
            .as_ref()
            .ok_or_else(|| DistributedError::Communication("no channel to peer".into()))?
            .send(values)
            .map_err(|_| DistributedError::Communication("peer rank stopped".into()))
    }

    fn receive(&self, peer: usize) -> DistributedResult<Vec<f64>> {
        self.receivers[peer]
            .as_ref()
            .ok_or_else(|| DistributedError::Communication("no channel from peer".into()))?
            .recv()
            .map_err(|_| DistributedError::Communication("peer rank stopped".into()))
    }
}

impl Communicator for ThreadCommunicator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn exchange(
        &mut self,
        peers: &[usize],
        sends: Vec<Vec<f64>>,
    ) -> DistributedResult<Vec<Vec<f64>>> {
        for (&peer, values) in peers.iter().zip(sends) {
            self.send(peer, values)?;
        }
        peers.iter().map(|&peer| self.receive(peer)).collect()
    }

    fn allgather(&mut self, local: &[f64]) -> DistributedResult<Vec<Vec<f64>>> {
        for peer in 0..self.size {
            if peer != self.rank {
                self.send(peer, local.to_vec())?;
            }
        }
        (0..self.size)
            .map(|peer| {
                if peer == self.rank {
                    Ok(local.to_vec())
                } else {
                    self.receive(peer)
                }
            })
            .collect()
    }
}

const HELLO: u64 = 0x524d_4444_5241_4e4b; // "RMDDRANK"

/// Ranks running as separate processes, possibly on separate machines, joined by a full TCP
/// mesh. Peer buffers are binary little-endian `f64` frames.
pub struct TcpCommunicator {
    rank: usize,
    size: usize,
    readers: Vec<Option<BufReader<TcpStream>>>,
    writers: Vec<Option<BufWriter<TcpStream>>>,
}

/// Coordinator side of the TCP rendezvous.
pub struct TcpRendezvous {
    listener: TcpListener,
    token: u64,
}

impl TcpRendezvous {
    pub fn bind(address: SocketAddr, token: u64) -> DistributedResult<Self> {
        Ok(Self {
            listener: TcpListener::bind(address)?,
            token,
        })
    }

    pub fn address(&self) -> DistributedResult<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Accept every rank, publish the peer table and return each rank's control stream.
    pub fn accept_ranks(&self, ranks: usize) -> DistributedResult<Vec<TcpStream>> {
        let mut streams = (0..ranks).map(|_| None).collect::<Vec<Option<TcpStream>>>();
        let mut addresses = vec![String::new(); ranks];
        for _ in 0..ranks {
            let (mut stream, _) = self.listener.accept()?;
            stream.set_nodelay(true)?;
            let magic = read_u64(&mut stream)?;
            let token = read_u64(&mut stream)?;
            let rank = read_u64(&mut stream)? as usize;
            let address = String::from_utf8(read_bytes(&mut stream)?)
                .map_err(|error| DistributedError::Communication(error.to_string()))?;
            if magic != HELLO || token != self.token || rank >= ranks || streams[rank].is_some() {
                return Err(DistributedError::Communication(
                    "rank rendezvous received an invalid hello".into(),
                ));
            }
            addresses[rank] = address;
            streams[rank] = Some(stream);
        }
        let table = addresses.join("\n").into_bytes();
        let mut streams = streams
            .into_iter()
            .map(|stream| stream.expect("every rank connected"))
            .collect::<Vec<_>>();
        for stream in &mut streams {
            write_bytes(stream, &table)?;
            stream.flush()?;
        }
        Ok(streams)
    }
}

impl TcpCommunicator {
    /// Join the mesh: announce this rank to the coordinator, then connect to lower ranks and
    /// accept higher ranks. Returns the communicator and the coordinator control stream.
    pub fn connect(
        coordinator: SocketAddr,
        token: u64,
        rank: usize,
        size: usize,
        bind_host: IpAddr,
    ) -> DistributedResult<(Self, TcpStream)> {
        let listener = TcpListener::bind(SocketAddr::new(bind_host, 0))?;
        let own = listener.local_addr()?;
        let mut control = TcpStream::connect(coordinator)?;
        control.set_nodelay(true)?;
        write_u64(&mut control, HELLO)?;
        write_u64(&mut control, token)?;
        write_u64(&mut control, rank as u64)?;
        write_bytes(&mut control, own.to_string().as_bytes())?;
        control.flush()?;
        let table = String::from_utf8(read_bytes(&mut control)?)
            .map_err(|error| DistributedError::Communication(error.to_string()))?;
        let addresses = table
            .split('\n')
            .map(|address| {
                address
                    .parse::<SocketAddr>()
                    .map_err(|error| DistributedError::Communication(error.to_string()))
            })
            .collect::<DistributedResult<Vec<_>>>()?;
        if addresses.len() != size {
            return Err(DistributedError::Communication(
                "peer table differs from the rank count".into(),
            ));
        }
        let mut streams = (0..size).map(|_| None).collect::<Vec<Option<TcpStream>>>();
        for (peer, address) in addresses.iter().enumerate().take(rank) {
            let mut stream = TcpStream::connect(address)?;
            stream.set_nodelay(true)?;
            write_u64(&mut stream, token)?;
            write_u64(&mut stream, rank as u64)?;
            stream.flush()?;
            streams[peer] = Some(stream);
        }
        for _ in rank + 1..size {
            let (mut stream, _) = listener.accept()?;
            stream.set_nodelay(true)?;
            let peer_token = read_u64(&mut stream)?;
            let peer = read_u64(&mut stream)? as usize;
            if peer_token != token || peer <= rank || peer >= size || streams[peer].is_some() {
                return Err(DistributedError::Communication(
                    "peer connection has an invalid identity".into(),
                ));
            }
            streams[peer] = Some(stream);
        }
        let mut readers = Vec::with_capacity(size);
        let mut writers = Vec::with_capacity(size);
        for stream in streams {
            match stream {
                Some(stream) => {
                    let writer = stream.try_clone()?;
                    readers.push(Some(BufReader::with_capacity(1 << 20, stream)));
                    writers.push(Some(BufWriter::with_capacity(1 << 20, writer)));
                }
                None => {
                    readers.push(None);
                    writers.push(None);
                }
            }
        }
        Ok((
            Self {
                rank,
                size,
                readers,
                writers,
            },
            control,
        ))
    }

    fn transfer(
        &mut self,
        peers: &[usize],
        sends: Vec<Vec<f64>>,
    ) -> DistributedResult<Vec<Vec<f64>>> {
        let mut outgoing = (0..self.size)
            .map(|_| None)
            .collect::<Vec<Option<Vec<f64>>>>();
        for (&peer, values) in peers.iter().zip(sends) {
            if outgoing[peer].replace(values).is_some() {
                return Err(DistributedError::Communication(
                    "one exchange sends twice to a peer".into(),
                ));
            }
        }
        let writers = &mut self.writers;
        let readers = &mut self.readers;
        std::thread::scope(|scope| {
            // One writer per peer: a writer only ever waits on its own peer's reader, so large
            // frames cannot deadlock the mesh.
            let handles = writers
                .iter_mut()
                .zip(outgoing)
                .filter_map(|(writer, values)| values.map(|values| (writer, values)))
                .map(|(writer, values)| {
                    scope.spawn(move || -> DistributedResult<()> {
                        let stream = writer.as_mut().ok_or_else(|| {
                            DistributedError::Communication("no connection to peer".into())
                        })?;
                        write_f64s(stream, &values)?;
                        stream.flush()?;
                        Ok(())
                    })
                })
                .collect::<Vec<_>>();
            let received = peers
                .iter()
                .map(|&peer| {
                    let stream = readers[peer].as_mut().ok_or_else(|| {
                        DistributedError::Communication("no connection from peer".into())
                    })?;
                    read_f64s(stream)
                })
                .collect::<DistributedResult<Vec<_>>>();
            for handle in handles {
                handle.join().map_err(|_| {
                    DistributedError::Communication("peer writer panicked".into())
                })??;
            }
            received
        })
    }
}

impl Communicator for TcpCommunicator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn exchange(
        &mut self,
        peers: &[usize],
        sends: Vec<Vec<f64>>,
    ) -> DistributedResult<Vec<Vec<f64>>> {
        if peers.len() != sends.len() || peers.contains(&self.rank) {
            return Err(DistributedError::Communication(
                "exchange lists are inconsistent".into(),
            ));
        }
        self.transfer(peers, sends)
    }

    fn allgather(&mut self, local: &[f64]) -> DistributedResult<Vec<Vec<f64>>> {
        let peers = (0..self.size)
            .filter(|&peer| peer != self.rank)
            .collect::<Vec<_>>();
        let sends = peers.iter().map(|_| local.to_vec()).collect();
        let mut received = self.transfer(&peers, sends)?.into_iter();
        Ok((0..self.size)
            .map(|peer| {
                if peer == self.rank {
                    local.to_vec()
                } else {
                    received.next().expect("one buffer per peer")
                }
            })
            .collect())
    }
}

pub(crate) fn write_u64(stream: &mut impl Write, value: u64) -> DistributedResult<()> {
    stream.write_all(&value.to_le_bytes())?;
    Ok(())
}

pub(crate) fn read_u64(stream: &mut impl Read) -> DistributedResult<u64> {
    let mut bytes = [0u8; 8];
    stream.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// Length-prefixed byte frame.
pub fn write_bytes(stream: &mut impl Write, bytes: &[u8]) -> DistributedResult<()> {
    write_u64(stream, bytes.len() as u64)?;
    stream.write_all(bytes)?;
    Ok(())
}

pub fn read_bytes(stream: &mut impl Read) -> DistributedResult<Vec<u8>> {
    let length = read_u64(stream)?;
    if length > (1 << 40) {
        return Err(DistributedError::Communication(
            "frame exceeds the transfer limit".into(),
        ));
    }
    let mut bytes = vec![0u8; length as usize];
    stream.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn write_f64s(stream: &mut impl Write, values: &[f64]) -> DistributedResult<()> {
    write_u64(stream, values.len() as u64)?;
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    stream.write_all(&bytes)?;
    Ok(())
}

fn read_f64s(stream: &mut impl Read) -> DistributedResult<Vec<f64>> {
    let count = read_u64(stream)? as usize;
    let mut bytes = vec![0u8; count * 8];
    stream.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight bytes")))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcp_mesh_exchanges_and_reduces() {
        let rendezvous = TcpRendezvous::bind("127.0.0.1:0".parse().unwrap(), 17).unwrap();
        let address = rendezvous.address().unwrap();
        let size = 3;
        let workers = (0..size)
            .map(|rank| {
                std::thread::spawn(move || {
                    let (mut comm, _control) = TcpCommunicator::connect(
                        address,
                        17,
                        rank,
                        size,
                        "127.0.0.1".parse().unwrap(),
                    )
                    .unwrap();
                    let mut values = vec![rank as f64, 1.0];
                    comm.allreduce_sum(&mut values).unwrap();
                    let next = (rank + 1) % size;
                    let previous = (rank + size - 1) % size;
                    let received = comm
                        .exchange(
                            &[next, previous],
                            vec![vec![rank as f64; 2], vec![-(rank as f64)]],
                        )
                        .unwrap();
                    (values, received)
                })
            })
            .collect::<Vec<_>>();
        let _controls = rendezvous.accept_ranks(size).unwrap();
        for (rank, worker) in workers.into_iter().enumerate() {
            let (values, received) = worker.join().unwrap();
            assert_eq!(values, vec![3.0, 3.0]);
            let previous = (rank + size - 1) % size;
            let next = (rank + 1) % size;
            assert_eq!(received[0], vec![-(next as f64)]);
            assert_eq!(received[1], vec![previous as f64; 2]);
        }
    }
}
