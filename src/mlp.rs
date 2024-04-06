use std::fmt::{Debug, Formatter};
use std::iter::zip;

use rand::distributions::{DistIter, Distribution};
use rand::Rng;

use crate::linear::Linear;
use crate::nd_tensor::NDTensor;

pub struct MLP {
    layers: Vec<Linear>,
}

impl Debug for MLP {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MLP {{\n{}\n}}",
               self.layers.iter()
                   .map(|l| format!("{}", l))
                   .collect::<Vec<_>>()
                   .join("\n")
        )
    }
}

impl MLP {
    pub fn new<D: Distribution<f32>, R: Rng>(input_size: usize, layer_sizes: Vec<usize>, rng: &mut DistIter<D, R, f32>) -> Self {
        let layers = layer_sizes.iter()
            .enumerate()
            .map(|(i, &cur_size)| {
                let prev_size = if i == 0 { input_size } else { layer_sizes[i - 1] };
                Linear::new(prev_size, cur_size, rng)
            })
            .collect();


        Self {
            layers
        }
    }

    pub fn forward(&self, x: &NDTensor<f32>) -> NDTensor<f32> {
        let mut x = x.clone();
        self.layers.iter()
            .for_each(|l|
                x = l.forward(&x)
            );

        x
    }

    pub fn backward(&mut self, x: &NDTensor<f32>, grad: &NDTensor<f32>) {
        // We need to compute the forward pass and accumulate the inputs
        // todo: do this during forward pass
        let mut xs = vec![x.clone()];

        self.layers[..self.layers.len()].iter()
            .enumerate()
            .for_each(|(i, l)| {
                xs.push(l.forward(&xs[i]))
            });

        // Go backwards and compute the gradient
        let mut grad = grad.clone();

        zip(self.layers.iter_mut(), xs.iter()).rev()
            .for_each(|(l, x)| {
                // Compute gradient for this layer
                let (gw, gb, gx) = l.gradient(&x, &grad);
                grad = gx;

                // Update weights
                l.update_weights(&gw, &gb)
            });
    }
}