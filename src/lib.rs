mod linear;
mod loss;
mod mlp;
mod nd_tensor;
mod utils;


#[cfg(test)]
mod tests {
    use rand::distributions::Standard;
    use rand::prelude::*;

    use crate::linear::Linear;
    use crate::loss::SumSquaredError;
    use crate::mlp::MLP;
    use crate::nd_tensor::NDTensor;

    fn get_dataset() -> (NDTensor<f32>, NDTensor<f32>) {
        let mut rng = StdRng::from_entropy().sample_iter(Standard);

        let inputs = NDTensor::from_rng(&mut rng, &vec![10, 2]);

        // Sum
        let gt = inputs.sum_axis(1);

        // Add some noise
        let noise = &(&NDTensor::from_rng(&mut rng, &gt.shape) - &0.5) * &0.05;
        let gt = &gt + &noise;


        (gt, inputs)
    }

    #[test]
    fn linear() {
        let mut rng = StdRng::from_entropy().sample_iter(Standard);
        let mut linear = Linear::new(2, 1, &mut rng);
        println!("{:?}", linear);


        let (gt, x) = get_dataset();
        println!("gt:\n{gt}");
        println!("inputs:\n{x}");

        let learning_rate = 0.001;

        println!("{:?}", linear);

        let epochs = 100;
        for i in 0..epochs {
            // Forward pass
            let y = linear.forward(&x);
            let loss = SumSquaredError::forward(&gt, &y);
            println!("epoch: {i}, loss {} gt {:?} pred {:?}", loss, gt, y);

            // Compute gradient
            let gradient = SumSquaredError::gradient(&gt, &y);
            let gradient = &gradient * &learning_rate;
            let (gw, gb, gx) = linear.gradient(&x, &gradient);

            // Step
            linear.update_weights(&gw, &gb);
        }
    }

    #[test]
    fn mlp() {
        let mut rng = StdRng::from_entropy().sample_iter(Standard);
        let mut mlp = MLP::new(2, vec![16, 16, 1], &mut rng);
        println!("{:?}", mlp);


        let (gt, x) = get_dataset();
        println!("gt:\n{gt}");
        println!("inputs:\n{x}");


        let learning_rate = 1e-1;
        let epochs = 100;

        for i in 0..epochs {
            // Forward pass
            let y = mlp.forward(&x);
            let loss = SumSquaredError::forward(&gt, &y);
            println!("epoch: {i}, loss {} gt {:?} pred {:?}", loss, gt, y);

            // Compute gradient
            let mut gradient = SumSquaredError::gradient(&gt, &y);
            gradient = &gradient * &learning_rate;

            // Step backwards
            mlp.backward(&x, &gradient);
        }
        println!("{:?}", mlp);
    }
}
