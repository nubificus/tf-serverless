use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use image::DynamicImage;
use log::{debug, info};
use serde::Serialize;
use tensorflow::{
    Code, Graph, SavedModelBundle, Session, SessionOptions, SessionRunArgs, Status, Tensor,
};

pub struct Timer {
    name: String,
    tstamp: Option<DateTime<Utc>>,
    duration: Option<Duration>,
}

impl Timer {
    /// Create a new timer
    pub fn new(name: &str) -> Self {
        Timer {
            name: name.to_owned(),
            tstamp: None,
            duration: None,
        }
    }

    pub fn new_start(name: &str) -> Self {
        let mut t = Timer::new(name);
        t.start();
        t
    }

    /// Start the timer
    pub fn start(&mut self) {
        info!("{}: starting", self.name);

        self.tstamp = Some(Utc::now());
        self.duration = None;
    }

    /// Stop the timer
    pub fn stop(&mut self) {
        match self.tstamp {
            None => debug!("{}: not running!", self.name),
            Some(tstamp) => {
                let d = Utc::now() - tstamp;

                self.duration = Some(d);
                self.tstamp = None;
                info!("{} duration: {} msec", self.name, d.num_milliseconds());
            }
        }
    }

    /// Get duration in milliseconds
    fn duration(&self) -> i64 {
        match self.duration {
            None => 0,
            Some(dur) => dur.num_milliseconds(),
        }
    }
}

pub struct ImageClassifier {
    /// TensorFlow model graph
    graph: Graph,

    /// TensorFlow session
    session: Session,

    /// Tags translation file
    tags: PathBuf,
}

#[derive(Default, Serialize)]
pub struct Classification {
    /// Classification tag of the image
    tag: String,

    /// Classification probability
    probability: f32,

    /// Time spent fetching image from URL
    time_url_fetch: i64,

    /// Time spent loading image in memory
    time_image_load: i64,

    /// Time resizing image
    time_image_resize: i64,

    /// Time spent on running session
    time_session_run: i64,
}

impl ImageClassifier {
    pub fn new(export_dir: &Path, tags_path: &Path) -> tensorflow::Result<Self> {
        let mut t = Timer::new_start("Loading session");

        let mut graph = Graph::new();
        let session =
            SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?
                .session;

        t.stop();

        Ok(ImageClassifier {
            graph,
            session,
            tags: tags_path.to_path_buf(),
        })
    }

    fn get_tag(&self, tensor: Tensor<f32>) -> tensorflow::Result<Classification> {
        let file = File::open(self.tags.clone())
            .map_err(|_| Status::new_set_lossy(Code::NotFound, "Could not open tags file"))?;

        let mut tags = BufReader::new(file).lines();

        let best = tensor
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        Ok(Classification {
            tag: tags.nth(best.0).unwrap().unwrap(),
            probability: *best.1,
            ..Default::default()
        })
    }

    pub fn run(&self, image: &[f32]) -> tensorflow::Result<Classification> {
        let mut t = Timer::new_start("Running session");

        let input = Tensor::new(&[1, 224, 224, 3])
            .with_values(&image)
            .expect("Bad image size");

        let mut args = SessionRunArgs::new();

        args.add_feed(
            &self
                .graph
                .operation_by_name_required("serving_default_input_1")?,
            0,
            &input,
        );

        let result = args.request_fetch(
            &self
                .graph
                .operation_by_name_required("StatefulPartitionedCall")?,
            0,
        );

        self.session.run(&mut args)?;
        let output = args.fetch(result)?;

        t.stop();

        let mut classification = self.get_tag(output)?;
        classification.time_session_run = t.duration();

        Ok(classification)
    }

    pub fn classify(&self, image: &DynamicImage) -> tensorflow::Result<Classification> {
        let mut t = Timer::new_start("Resizing image");

        let rgb = image.to_rgb();

        let resized =
            image::imageops::resize(&rgb, 224, 224, image::imageops::FilterType::Triangle);

        let raw_image: Vec<f32> = resized
            .into_raw()
            .iter()
            .map(|x| *x as f32 / 255f32)
            .collect();

        t.stop();

        let mut classification = self.run(&raw_image)?;
        classification.time_image_resize = t.duration();

        Ok(classification)
    }

    pub fn classify_from_raw(&self, data: &[u8]) -> tensorflow::Result<Classification> {
        let mut t = Timer::new_start("Load image from memory");

        let image = image::load_from_memory(&data).map_err(|_| {
            Status::new_set_lossy(Code::InvalidArgument, "Could create image from raw data")
        })?;

        t.stop();

        let mut classification = self.classify(&image)?;
        classification.time_image_load = t.duration();

        Ok(classification)
    }

    pub fn classify_from_url(&self, url: &str) -> tensorflow::Result<Classification> {
        let mut t = Timer::new_start(&format!("Fetching image from {}", url));

        let mut resp =
            reqwest::get(url).map_err(|_| Status::new_set_lossy(Code::NotFound, "Invalid URL"))?;

        let mut buf: Vec<u8> = vec![];
        resp.copy_to(&mut buf)
            .map_err(|_| Status::new_set_lossy(Code::DataLoss, "Could not read image from URL"))?;

        t.stop();

        let mut classification = self.classify_from_raw(&buf)?;
        classification.time_url_fetch = t.duration();

        Ok(classification)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
