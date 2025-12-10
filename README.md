## Self-Healing Vector DB (Rust + HNSW + SQLite + Axum)

This project is a **self-healing vector database** prototype built with:

- **Rust** for performance and safety
- **`hnsw_rs`** for approximate nearest neighbor search
- **SQLite** for durable vector storage
- **Axum** for a simple HTTP API
- **ONNX or external embedding API** (planned) for text-to-vector embeddings

### High-Level Architecture

- **`src/index.rs`**: HNSW index wrapper (`HnswIndex`) using `hnsw_rs`.
- **`src/storage.rs`**: `SqliteVectorStore` – raw vectors + IDs in SQLite (source of truth).
- **`src/engine.rs`**: `SelfHealingVectorDb` – wires index + storage + (optional) embedder.
- **`src/health.rs`**: basic health report over the index.
- **`src/embeddings.rs`**: `Embedder` trait and a dummy implementation (swap in ONNX/API later).
- **`src/main.rs`**: Axum server exposing `/add`, `/search`, `/health`.

On startup, the engine:

- opens the SQLite store
- loads all stored vectors
- rebuilds the in-memory HNSW index from storage (**self-healing bootstrap**).

### Running the server

```bash
cargo run --bin self_healing_vector_db_server
```

The server listens on `http://127.0.0.1:3000` with:

- `POST /add` – add vectors by ID
- `POST /search` – search nearest neighbors
- `GET /health` – simple health report

### If you don't have Rust installed

1. **Install Rust (using rustup)**

On macOS or Linux:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then restart your terminal so `cargo` is on your `PATH`.  
You can confirm with:

```bash
cargo --version
```

On Windows, use the installer from the official Rust site.

2. **Clone and run**

```bash
git clone https://github.com/<your-username>/self-healing-vector-db.git
cd self-healing-vector-db
cargo run --bin self_healing_vector_db_server
```

You can now use `curl` (or any HTTP client) to call `/add`, `/search`, and `/health` as shown above.

