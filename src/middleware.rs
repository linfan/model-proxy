use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use tracing::info;

pub async fn logging(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();

    info!("Started {} {}", method, uri);
    let start = std::time::Instant::now();

    let response = next.run(req).await;

    let status = response.status();
    let elapsed = start.elapsed();

    info!("Completed {} {} with status {} in {:?}", method, uri, status, elapsed);

    response
}
