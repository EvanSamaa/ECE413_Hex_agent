/// Python bindings for the hex game and MCTS logic

use pyo3::prelude::*;
use numpy::{PyArrayDyn, IntoPyArray};
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use ordered_float::OrderedFloat;

use mcts_lib::{search, hex, Game};

use ndarray::prelude::*;

#[pyclass]
#[derive(Clone)]
struct PyHex {
    game: hex::Hex,
}

#[pymethods]
impl PyHex {
    #[new]
    fn new(n: usize) -> Self {
        PyHex{
            game: hex::Hex::new(n),
        }
    }

    fn next_state(&self, action: u32) -> Self {
        PyHex { game: self.game.next_state(action) }
    }

    fn terminal_value(&self) -> Option<f32> {
        self.game.terminal_value()
    }

    fn valid_actions(&self) -> Vec<u32> {
        self.game.valid_actions().collect()
    }

    fn to_string(&self) -> String {
        format!("{}", self.game)
    }

    fn to_array<'py>(&self, py: Python<'py>) -> &'py PyArrayDyn<f32> {
        self.game.to_array().into_dyn().into_pyarray(py)
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}

#[pyclass]
struct PySearch {
    tree: search::SearchTree<hex::Hex>,
}

#[pymethods]
impl PySearch {
    #[new]
    fn new() -> Self {
        PySearch {
            tree: search::SearchTree::new(),
        }
    }

    fn get_action<'py>(&mut self, py: Python<'py>, game: PyHex, iterations: usize, value_fn: Option<PyObject>) -> u32 {
        let model = |g: hex::Hex| {
            if let Some(value_fn) = &value_fn {
                let result = value_fn.call(py, (g.to_array().into_dyn().into_pyarray(py),), None)
                    .expect("Failed calling value function");
                result.extract(py).expect("Failed extracting from value function result")
            } else {
                search::rollout(g)
            }
        };
        
        for _ in 0..iterations {
            self.tree.search(&game.game, &model);
        }
        
        let action_prob = self.tree.action_prob(&game.game);
        let action = (0..action_prob.len()).max_by_key(|&i| OrderedFloat(action_prob[i])).unwrap() as u32;
        action
    }

    fn self_play<'py>(&mut self, py: Python<'py>, size: usize, iterations: usize, temperature: f32, value_fn: Option<PyObject>)
        -> (&'py PyArrayDyn<f32>, &'py PyArrayDyn<f32>, &'py PyArrayDyn<f32>)
    {
        let (states, values, action_probs) = if let Some(value_fn) = value_fn {
            let model = |g: hex::Hex| {
                value_fn.call(py, (g.to_array().into_dyn().into_pyarray(py),), None)
                    .expect("Failed calling value function")
                    .extract(py)
                    .expect("Failed extracting float from value function result")
            };
            self_play(&mut self.tree, size, iterations, temperature, model)
        } else {
            self_play(&mut self.tree, size, iterations, temperature, search::rollout)
        };
        (states.into_pyarray(py), values.into_pyarray(py), action_probs.into_pyarray(py))
    }
}

fn self_play<M: Fn(hex::Hex) -> (f32, Vec<f32>)>(tree: &mut search::SearchTree<hex::Hex>, size: usize, iterations: usize, temperature: f32, model: M)
    -> (ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)
{
    let mut state = hex::Hex::new(size);
    let mut states = Vec::new();
    let mut action_probs: Vec<f32> = Vec::new();
    let mut value = -1.;

    while state.terminal_value().is_none() {
        for _ in 0..iterations {
            tree.search(&state, &model);
        }

        // Sample an action according to the action probabilities
        let action_prob = tree.action_prob(&state);
        let action_prob_perturbed = action_prob.iter().map(|p| p.powf(1. / temperature));
        let dist = WeightedIndex::new(action_prob_perturbed).unwrap();
        let action = dist.sample(&mut thread_rng()) as u32;

        states.push(state.to_array());
        action_probs.extend(action_prob);

        state = state.next_state(action);
        value *= -1.;
    }

    let state_views: Vec<_> = states.iter().map(|arr| arr.view()).collect();
    let value_targets: Vec<_> = (0..states.len()).map(|i| if i % 2 == 0 { value } else { -value }).collect();

    let states = ndarray::stack(Axis(0), &state_views).unwrap().into_dyn();
    let action_probs = Array::from_shape_vec((state_views.len(), state.action_count()), action_probs)
        .unwrap().into_dyn();
    let values = Array::from(value_targets).into_dyn();

    return (states, values, action_probs)
}

#[pymodule]
fn mcts_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyHex>()?;
    m.add_class::<PySearch>()?;
    Ok(())
}

