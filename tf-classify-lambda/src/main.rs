use lambda_runtime::{handler_fn, Context};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tf_serve::ImageClassifier;

type Error = Box<dyn std::error::Error + Sync + Send + 'static>;

#[derive(Deserialize)]
struct Request {
    url: String,
}

#[derive(Serialize)]
struct Response {
    tag: String,
    probability: f32,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    env_logger::init();

    let export_dir = PathBuf::from("/mnt/libraries/resnet50");
    let tags_path = PathBuf::from("/mnt/libraries/resnet50/ImageNetLabels.txt");
    let classifier = ImageClassifier::new(&export_dir, &tags_path)?;

    let classifier_ref = &classifier;

    let func = handler_fn(move |event: Request, ctx: Context| async move {
        handler(event, ctx, classifier_ref)
    });

    lambda_runtime::run(func).await?;

    Ok(())
}

fn handler(event: Request, _: Context, classifier: &ImageClassifier) -> Result<Response, Error> {
    let (tag, probability) = classifier.classify_from_url(&event.url)?;

    Ok(Response { tag, probability })
}
