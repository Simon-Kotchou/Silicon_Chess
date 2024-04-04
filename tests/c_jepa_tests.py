import unittest
import torch
from Net.c_jepa import square_masking  

class TestChessBoardMasking(unittest.TestCase):

    def setUp(self):
        self.board_size = 8  # Standard chess board size

    def test_mask_shapes(self):
        """Test that the generated masks have the correct shapes."""
        context_mask, target_masks, _ = square_masking(self.board_size)
        self.assertEqual(context_mask.shape, (self.board_size, self.board_size))
        for target_mask in target_masks:
            self.assertEqual(target_mask.shape, (self.board_size, self.board_size))

    def test_mask_values(self):
        """Test that masks only contain binary values (0s and 1s)."""
        context_mask, target_masks, _ = square_masking(self.board_size)
        self.assertTrue(torch.all((context_mask == 0) | (context_mask == 1)))
        for target_mask in target_masks:
            self.assertTrue(torch.all((target_mask == 0) | (target_mask == 1)))

    def test_target_coverage(self):
        """Test that the number of targets is within the expected range based on target_scale."""
        _, target_masks, _ = square_masking(self.board_size, target_scale=(0.1, 0.2))
        total_targets = sum(torch.sum(mask) for mask in target_masks)
        min_expected_targets = self.board_size ** 2 * 0.1
        max_expected_targets = self.board_size ** 2 * 0.2
        self.assertTrue(min_expected_targets <= total_targets <= max_expected_targets)

    def test_context_integrity(self):
        """Test that the context mask excludes target squares."""
        context_mask, target_masks, _ = square_masking(self.board_size)
        for target_mask in target_masks:
            self.assertTrue(torch.all((context_mask + target_mask) <= 1))

    def test_non_overlapping_targets(self):
        """Test that target masks do not overlap."""
        _, target_masks, _ = square_masking(self.board_size, num_targets=4)
        combined_mask = torch.sum(torch.stack(target_masks), dim=0)
        self.assertTrue(torch.all(combined_mask <= 1))

if __name__ == '__main__':
    unittest.main()