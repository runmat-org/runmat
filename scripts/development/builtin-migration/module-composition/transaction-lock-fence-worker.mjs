import net from "node:net";
import { parentPort, workerData } from "node:worker_threads";

const state = new Int32Array(workerData.state);
let completed = false;

if (workerData.mode === "serve") serve();
else if (workerData.mode === "probe") probe();
else complete(-1);

function serve() {
  const server = net.createServer((socket) => socket.end(`${workerData.token}\n`));
  server.on("error", () => complete(-1));
  server.listen({ host: "127.0.0.1", port: 0, exclusive: true }, () => {
    Atomics.store(state, 1, server.address().port);
    complete(1);
  });
  parentPort.on("message", (message) => {
    if (message === "close") server.close(() => complete(2));
  });
}

function probe() {
  const socket = net.createConnection({ host: "127.0.0.1", port: workerData.port });
  let response = "";
  socket.setEncoding("utf8");
  socket.setTimeout(workerData.timeoutMs);
  socket.on("data", (value) => { response += value; if (response.length > 128) socket.destroy(); });
  socket.on("end", () => complete(response === `${workerData.token}\n` ? 1 : 2));
  socket.on("timeout", () => { socket.destroy(); complete(-1); });
  socket.on("error", () => complete(2));
}

function complete(value) {
  if (workerData.mode === "probe" && completed) return;
  if (workerData.mode === "serve" && Atomics.load(state, 0) === 2) return;
  completed = true;
  Atomics.store(state, 0, value);
  Atomics.notify(state, 0);
}
