export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center px-6 py-12">
      <div className="w-full max-w-3xl">
        <header className="mb-10 border-b border-slate-200 pb-6">
          <h1 className="text-3xl font-semibold tracking-tight text-red-600">
            Self-Healing Vector DB
          </h1>
          <p className="mt-3 text-sm text-red-500">
            A Rust + HNSW + SQLite experimental vector database that can detect
            issues and rebuild itself from durable storage.
          </p>
        </header>

        <section className="space-y-6 text-sm leading-relaxed text-black">
          <p>
            This project explores what it would look like if a vector database
            could actively maintain its own health. Instead of silently
            degrading over time, it keeps raw vectors in SQLite and uses an
            in-memory HNSW index for fast approximate nearest-neighbor search.
          </p>

          <p>
            On startup, the engine bootstraps the HNSW index from SQLite. If
            the index is ever corrupted or needs to be rebuilt, the system can
            reconstruct it from the stored source of truth. Over time, the goal
            is to add recall testing, drift detection, and automatic
            self-repair.
          </p>

          <p>
            The backend is written in Rust using Axum for the HTTP API, hnsw_rs
            for the ANN index, and SQLite for storage. A future iteration will
            plug in an ONNX model or external embedding API so you can index raw
            text instead of precomputed vectors.
          </p>

          <p>
            This page is just a tiny log of the journey: from a simple
            prototype to a more serious, self-aware vector store that measures
            and improves its own quality.
          </p>
        </section>
      </div>
    </main>
  );
}


