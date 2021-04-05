use std::collections::HashMap;
use std::hash::Hash;
use rand::Rng;
use ordered_float::OrderedFloat;

use crate::Game;

pub struct SearchTree<G: Hash> {
    nodes: HashMap<G, Node>,
}

enum Node {
    Terminal {
        value: f32,
    },
    Inner {
        n: f32,
        stats: HashMap<u32, Stats>,
    },
}

struct Stats {
    v: f32,
    n: f32,
}

impl <G: Eq + Hash + Game + Clone> SearchTree<G> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Action probablity estimate from the given state based on previous search iterations
    pub fn action_prob(&self, s: &G) -> Vec<f32> {
        let node = self.nodes.get(s);
        match node {
            Some(Node::Inner{ stats, .. }) => {
                let mut probs = vec![0.; s.action_count()];
                let actions = s.valid_actions();
                for a in actions {
                    probs[a as usize] += stats.get(&a).map(|s| s.n).unwrap_or(0.);
                }

                let sum: f32 = probs.iter().sum();
                if sum == 0. {
                    panic!("The state has not been searched")
                }
                for i in 0..probs.len() {
                    probs[i] /= sum;
                }
                probs
            },
            Some(Node::Terminal{ .. }) => panic!("The state is terminal"),
            None => panic!("The state has not been searched"),
        }
    }
    
    /// Runs an iteration of Monte-Carlo Tree Search
    pub fn search<M: Fn(G) -> f32>(&mut self, s: &G, model: M) -> f32 {
        match self.nodes.get(&s) {
            Some(Node::Terminal{ value }) => *value,
            Some(Node::Inner{ .. }) => self.search_descend(s, model),
            None => {
                let node = match s.terminal_value() {
                    Some(value) => Node::Terminal{ value },
                    None => Node::Inner{
                        n: 0.,
                        stats: HashMap::new(),
                    },
                };
                let value = match &node {
                    Node::Terminal{ value } => *value,
                    Node::Inner{ .. } => model(s.clone()),
                };
                self.nodes.insert(s.clone(), node);
                value
            },
        }
    }

    fn search_descend<M: Fn(G) -> f32>(&mut self, s: &G, model: M) -> f32 {
        let cpuct = 2.0_f32.sqrt();
        let action = match self.nodes.get(s) {
            Some(Node::Inner{ n: n_s, stats }) => {
                // Pick the action with the highest upper confidence bound
                let actions = s.valid_actions();
                let action_count = s.action_count() as f32;
                actions.max_by_key(|a|
                    match stats.get(a) {
                        Some(Stats { v, n: n_sa }) =>
                            OrderedFloat((*v / *n_sa) + cpuct / action_count * ( *n_s / *n_sa ).sqrt()),
                        None =>
                            OrderedFloat(std::f32::INFINITY),
                    }
                ).unwrap()
            },
            _ => panic!("Descend must only be called on an inner node"),
        };

        let next_state = s.next_state(action);
        let v = self.search(&next_state, model);

        if let Some(Node::Inner{ n, stats }) = self.nodes.get_mut(s) {
            let stats = stats.entry(action).or_insert(Stats{ v: 0., n: 0. });
            stats.v += -v;
            stats.n += 1.;
            *n += 1.;
        }
        
        -v
    }
}

fn select_action<G: Game>(state: &G) -> u32 {
    let mut rng = rand::thread_rng();
    let actions_count = state.valid_actions_count();
    state.valid_actions().nth(rng.gen_range(0..actions_count)).unwrap()
}

pub fn rollout2<G: Game>(_: G) -> f32 {
    let mut rng = rand::thread_rng();
    rng.gen_range(-1.0..1.0)
}

pub fn rollout<G: Game>(state: G) -> f32 {
    match state.terminal_value() {
        Some(v) => v,
        None => -rollout(state.next_state(select_action(&state))),
    }
}

