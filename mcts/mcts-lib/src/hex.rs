use petgraph::unionfind::UnionFind;
use crate::Game;

#[derive(Clone)]
pub struct Hex {
    n: usize,
    turn: usize,
    pieces: [Vec<bool>; 2],
    sets: [UnionFind<u16>; 2],
}

impl Hex {
    pub fn new(n: usize) -> Self {
        let n2 = n.pow(2);
        Hex {
            n,
            turn: 0,
            pieces: [vec![false; n2], vec![false; n2]],
            sets: [UnionFind::new(n2 + 2), UnionFind::new(n2 + 2)],
        }
    }
}

impl PartialEq for Hex {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n
            && self.turn == other.turn
            && self.pieces == other.pieces
    }
}

impl Eq for Hex {}

impl std::hash::Hash for Hex {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.n.hash(state);
        self.turn.hash(state);
        self.pieces.hash(state);
    }
}

impl Game for Hex {
    type ActionIter<'a> = impl Iterator<Item=u32> + 'a;

    fn action_count(&self) -> usize {
        self.n.pow(2)
    }

    fn valid_actions(&self) -> Self::ActionIter<'_> {
        (0..self.n.pow(2))
            .filter(move |&i| !self.pieces[0][i] && !self.pieces[1][i])
            .map(|i| i as u32)
    }

    fn next_state(&self, pos: u32) -> Self {
        let mut next = self.clone();

        let pos = pos as usize;
        let n = self.n as i32;
        let r = pos as i32 / n;
        let c = pos as i32 % n;

        next.pieces[next.turn][pos] = true;

        // Neighbors
        let ds: &[(i32, i32)] = &[(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)];
        for &(dy, dx) in ds {
            let y = r + dy;
            let x = c + dx;
            if y < 0 || y >= n || x < 0 || x >= n {
                continue
            }
            let i = y * n + x;
            if next.pieces[next.turn][i as usize] {
                next.sets[next.turn].union(pos as u16, i as u16);
            }
        }

        // Edges
        let e0 = (n as u16).pow(2);
        let e1 = e0 + 1;
        if next.turn == 0 {
            if r == 0 {
                next.sets[0].union(pos as u16, e0);
            } else if r == n - 1 {
                next.sets[0].union(pos as u16, e1);
            }
        } else if next.turn == 1 {
            if c == 0 {
                next.sets[1].union(pos as u16, e0);
            } else if c == n - 1 {
                next.sets[1].union(pos as u16, e1);
            }
        }

        next.turn = 1 - next.turn;
        next
    }

    fn terminal_value(&self) -> Option<f32> {
        let e0 = (self.n as u16).pow(2);
        let e1 = e0 + 1;
        if self.sets[0].equiv(e0, e1) {
            Some(if self.turn == 0 { 1. } else { -1. })
        } else if self.sets[1].equiv(e0, e1) {
            Some(if self.turn == 1 { 1. } else { -1. })
        } else {
            None
        }
    }

    fn to_array(&self) -> ndarray::Array3<f32> {
        // We always return the board from the point of view of the current player
        // So if turn = 1, we swap the layers and transpose the pieces
        let mut array = ndarray::Array3::zeros((2, self.n, self.n));
        for player in 0..=1 {
            let layer = if self.turn == 0 { player } else { 1 - player }; 
            for (index, &piece) in self.pieces[player].iter().enumerate() {
                if piece {
                    let row = if self.turn == 0 { index / self.n } else { index % self.n };
                    let col = if self.turn == 0 { index % self.n } else { index / self.n };
                    array[(layer, row, col)] = 1.;
                }
            }
        }
        array
    }
}

impl std::fmt::Display for Hex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.n;
        for i in 0..n {
            write!(f, "{}", " ".repeat(i))?;
            for j in 0..n {
                write!(f, "{} ", match (self.pieces[0][i * n + j], self.pieces[1][i * n + j]) {
                    (false, false) => "-",
                    (true, false) => "x",
                    (false, true) => "o",
                    _ => unimplemented!(),
                })?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

