use lambda_http::{
    handler,
    lambda_runtime::{self, Context, Error},
    IntoResponse, Request, Response,
};

use log::debug;
use std::path::PathBuf;
use tf_serve::ImageClassifier;

extern crate base64;
extern crate serde_json;

#[tokio::main]
async fn main() -> Result<(), Error> {
    env_logger::init();

    let export_dir = PathBuf::from("/mnt/libraries/resnet50");
    let tags_path = PathBuf::from("/mnt/libraries/resnet50/ImageNetLabels.txt");
    let classifier = ImageClassifier::new(&export_dir, &tags_path)?;

    debug!("Loaded model in memory");

    let classifier_ref = &classifier;

    let handler_closure = move |event: Request, ctx: Context| async move {
        handle_request(event, ctx, classifier_ref)
    };

    debug!("Dispatching handler");
    lambda_runtime::run(handler(handler_closure)).await?;

    Ok(())
}

fn handle_request(
    event: Request,
    _ctx: Context,
    classifier: &ImageClassifier,
) -> Result<impl IntoResponse, Error> {
    debug!("Inside handler");
    debug!("Received request: {:#?}", event);

    let mut t = tf_serve::Timer::new_start("Handling request");

    let body = event.body();

    let response = match classifier.classify_from_raw(body) {
        Err(err) => Response::builder()
            .body(format!("Classification failure: '{}'", err))
            .expect("Failed to render response"),
        Ok(classification) => Response::builder()
            .status(200)
            .body(serde_json::to_string(&classification)?)
            .expect("Failed to render response"),
    };

    t.stop();

    Ok(response)
}
