/// Simple array-based set using a boolean vector
/// Optimized for the case where elements are in range [0, size)
#[derive(Debug, Clone)]
pub struct SimpleSet {
    present: Vec<bool>,
    count: usize,
}

impl SimpleSet {
    pub fn new(size: usize) -> Self {
        Self {
            present: vec![false; size],
            count: 0,
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, key: usize) {
        if !self.present[key] {
            self.present[key] = true;
            self.count += 1;
        }
    }

    #[inline(always)]
    pub fn remove(&mut self, key: usize) {
        if self.present[key] {
            self.present[key] = false;
            self.count -= 1;
        }
    }

    #[inline(always)]
    pub fn contains(&self, key: usize) -> bool {
        self.present[key]
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Iterator over elements in the set
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.present
            .iter()
            .enumerate()
            .filter_map(|(i, &present)| if present { Some(i) } else { None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_set() {
        let mut set = SimpleSet::new(10);

        assert!(set.is_empty());

        set.insert(3);
        set.insert(7);
        set.insert(1);

        assert_eq!(set.len(), 3);
        assert!(set.contains(3));
        assert!(set.contains(7));
        assert!(!set.contains(5));

        set.remove(7);
        assert_eq!(set.len(), 2);
        assert!(!set.contains(7));

        let elements: Vec<usize> = set.iter().collect();
        assert_eq!(elements, vec![1, 3]);
    }
}
