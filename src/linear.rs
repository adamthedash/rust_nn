use std::fmt::{Display, Formatter};
use rand::distributions::DistIter;
use rand::prelude::*;

use crate::nd_tensor::NDTensor;

#[derive(Debug)]
pub struct Linear {
    // (i, n)
    w: NDTensor<f32>,
    // (1, n)
    b: NDTensor<f32>,
}

impl Linear {
    pub fn new<D: Distribution<f32>, R: Rng>(input_size: usize, units: usize, rng: &mut DistIter<D, R, f32>) -> Self {
        Self {
            w: &NDTensor::from_rng(rng, &vec![input_size, units]) / &(input_size as f32),
            b: NDTensor::from_rng(rng, &vec![1, units]),
        }
    }

    pub fn forward(&self, x: &NDTensor<f32>) -> NDTensor<f32> {
        assert_eq!(x.shape.len(), 2);  // (B, F)
        let mul = x.matmul(&self.w);

        &mul + &self.b
    }

    pub fn gradient(&self, x: &NDTensor<f32>, grad: &NDTensor<f32>) -> (NDTensor<f32>, NDTensor<f32>, NDTensor<f32>) {
        // dg/db (N,)
        let db = grad.sum_axis(0);

        // dg/dw (N, i)
        let dw = x.transpose().matmul(grad);

        // dg/dx (i,)
        let dx = grad.matmul(&self.w.transpose());

        (dw, db, dx)
    }

    pub fn update_weights(&mut self, dw: &NDTensor<f32>, db: &NDTensor<f32>) {
        assert_eq!(dw.shape, self.w.shape);
        assert_eq!(db.shape, self.b.shape);

        self.b = &self.b - db;
        self.w = &self.w - dw;
    }
}

impl Display for Linear {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Linear (\nw: {},\nb: {}\n)", self.w, self.b)
    }
}