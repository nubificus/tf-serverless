use std::error::Error;
use std::path::PathBuf;
use structopt::StructOpt;
use tf_serve::ImageClassifier;

extern crate serde_json;

use log::info;

#[derive(StructOpt, Debug)]
#[structopt(
    name = "tf-classify",
    about = "CLI app to perform image classification with TensorFlow"
)]
struct CmdArgs {
    #[structopt(help = "Export directory of TensorFlow SavedModel")]
    export_dir: String,

    #[structopt(help = "Path to tags translation file")]
    tags_path: String,

    #[structopt(help = "URL to fetch image from")]
    image_url: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let args = CmdArgs::from_args();

    let export_dir = PathBuf::from(args.export_dir);
    let tags_path = PathBuf::from(args.tags_path);

    let classifier = ImageClassifier::new(&export_dir, &tags_path)?;

    let classification = classifier.classify_from_url(&args.image_url)?;

    info!("{}", serde_json::to_string(&classification).unwrap());

    Ok(())
}
