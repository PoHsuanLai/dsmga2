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
    pub fn iter(&self) -> SimpleSetIter<'_> {
        SimpleSetIter {
            iter: self.present.iter().enumerate(),
        }
    }
}

/// Iterator over elements in a SimpleSet
pub struct SimpleSetIter<'a> {
    iter: std::iter::Enumerate<std::slice::Iter<'a, bool>>,
}

impl<'a> Iterator for SimpleSetIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .find_map(|(i, &present)| if present { Some(i) } else { None })
    }
}

impl<'a> IntoIterator for &'a SimpleSet {
    type Item = usize;
    type IntoIter = SimpleSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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

        // Test IntoIterator trait
        let elements_via_trait: Vec<usize> = (&set).into_iter().collect();
        assert_eq!(elements_via_trait, vec![1, 3]);

        // Test for loop (uses IntoIterator)
        let mut count = 0;
        for _ in &set {
            count += 1;
        }
        assert_eq!(count, 2);
    }
}
