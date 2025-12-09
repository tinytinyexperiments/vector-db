use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

use self_healing_vector_db::embeddings::DummyEmbedder;
use self_healing_vector_db::engine::{EngineConfig, SelfHealingVectorDb};

#[derive(Clone)]
struct AppState {
    engine: Arc<RwLock<SelfHealingVectorDb>>,
}

#[derive(Debug, Deserialize)]
struct AddRequest {
    ids: Vec<i64>,
    vectors: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    query: Vec<f32>,
    k: usize,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    ok: bool,
    reason: String,
    size: usize,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let dim = 384;
    let engine_cfg = EngineConfig {
        dim,
        storage_path: "data/vectors.sqlite".into(),
        hnsw_max_elements: 100_000,
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        hnsw_ef_search: 64,
    };

    let embedder = DummyEmbedder { dim };
    let engine = SelfHealingVectorDb::new(engine_cfg, Some(Arc::new(embedder)))
        .expect("failed to create engine");

    let state = AppState {
        engine: Arc::new(RwLock::new(engine)),
    };

    let app = Router::new()
        .route("/add", post(add_handler))
        .route("/search", post(search_handler))
        .route("/health", get(health_handler))
        .with_state(state);

    let addr: SocketAddr = "127.0.0.1:3000".parse().unwrap();
    tracing::info!("listening on http://{}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn add_handler(
    State(state): State<AppState>,
    Json(payload): Json<AddRequest>,
) -> Json<&'static str> {
    let mut engine = state.engine.write().await;
    let _ = engine.add_vectors(&payload.ids, &payload.vectors);
    Json("ok")
}

async fn search_handler(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> Json<serde_json::Value> {
    let engine = state.engine.read().await;
    let results = engine.search(&payload.query, payload.k).unwrap_or_default();
    Json(serde_json::json!(results))
}

async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    let engine = state.engine.read().await;
    let report = engine.health();
    Json(HealthResponse {
        ok: report.ok,
        reason: report.reason,
        size: report.size,
    })
}


