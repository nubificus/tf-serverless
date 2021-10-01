use hyper::server::conn::AddrStream;
use hyper::service::{make_service_fn, service_fn};
use hyper::{body, Body, Request, Response, Server};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tf_serve::ImageClassifier;

extern crate serde;

async fn handle(
    req: Request<Body>,
    classifier: Arc<ImageClassifier>,
) -> Result<Response<Body>, Infallible> {
    let raw = body::to_bytes(req.into_body()).await.unwrap();

    let response = match classifier.classify_from_raw(&raw) {
        Err(err) => Response::builder()
            .status(400)
            .body(Body::from(format!("Classification failure: '{}'", err))),
        Ok(result) => Response::builder()
            .status(200)
            .body(Body::from(serde_json::to_string(&result).unwrap())),
    };

    Ok(response.unwrap())
}

#[tokio::main]
async fn main() {
    let export_dir = PathBuf::from("/opt/resnet50");
    let tags_path = PathBuf::from("/opt/resnet50/ImageNetLabels.txt");

    let classifier = Arc::new(ImageClassifier::new(&export_dir, &tags_path).unwrap());

    // A `MakeService` that produces a `Service` to handle each connection.
    let make_service = make_service_fn(move |_conn: &AddrStream| {
        // We have to clone the context to share it with each invocation of
        // `make_service`. If your data doesn't implement `Clone` consider using
        // an `std::sync::Arc`.

        let class = Arc::clone(&classifier);

        // Create a `Service` for responding to the request.
        let service = service_fn(move |req| handle(req, class.clone()));

        // Return the service to hyper.
        async move { Ok::<_, Infallible>(service) }
    });

    // Run the server like above...
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));

    let server = Server::bind(&addr).serve(make_service);

    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
    }
}
