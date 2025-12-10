use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::State,
    http::{Request, StatusCode},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use self_healing_vector_db::{EngineConfig, SelfHealingVectorDb};
use tempfile::tempdir;
use tokio::sync::RwLock;
use tower::ServiceExt;

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

#[derive(Debug, Serialize, Deserialize)]
struct SearchResult {
    id: usize,
    distance: f32,
}

async fn add_handler(
    State(state): State<AppState>,
    Json(payload): Json<AddRequest>,
) -> StatusCode {
    let mut engine = state.engine.write().await;
    match engine.add_vectors(&payload.ids, &payload.vectors) {
        Ok(_) => StatusCode::OK,
        Err(_) => StatusCode::BAD_REQUEST,
    }
}

async fn search_handler(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> Json<Vec<SearchResult>> {
    let engine = state.engine.read().await;
    let results = engine
        .search(&payload.query, payload.k)
        .unwrap_or_default();
    let out = results
        .into_iter()
        .map(|r| SearchResult {
            id: r.id,
            distance: r.distance,
        })
        .collect();
    Json(out)
}

#[tokio::test]
async fn http_add_and_search_roundtrip() {
    let dim = 4;
    let tmp_dir = tempdir().expect("tempdir");
    let db_path: PathBuf = tmp_dir.path().join("vectors.sqlite");

    let cfg = EngineConfig {
        dim,
        storage_path: db_path,
        hnsw_max_elements: 10_000,
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        hnsw_ef_search: 64,
    };

    let engine =
        SelfHealingVectorDb::new(cfg, None).expect("engine created");

    let state = AppState {
        engine: Arc::new(RwLock::new(engine)),
    };

    let app = Router::new()
        .route("/add", post(add_handler))
        .route("/search", post(search_handler))
        .with_state(state);

    // Add two vectors via HTTP.
    let add_body = serde_json::json!({
        "ids": [1, 2],
        "vectors": [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0]
    })
    .to_string();

    let response = app
        .clone()
        .oneshot(
            Request::post("/add")
                .header("content-type", "application/json")
                .body(add_body.into())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Search via HTTP.
    let search_body = serde_json::json!({
        "query": [1.0, 0.0, 0.0, 0.0],
        "k": 2
    })
    .to_string();

    let response = app
        .oneshot(
            Request::post("/search")
                .header("content-type", "application/json")
                .body(search_body.into())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Vec<SearchResult> =
        serde_json::from_slice(&bytes).expect("valid JSON response");

    assert!(!body.is_empty(), "expected at least one search result");
    assert_eq!(body[0].id, 1, "nearest neighbor over HTTP should be id 1");
}


